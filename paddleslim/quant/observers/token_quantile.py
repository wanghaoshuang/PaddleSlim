# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
from paddle import _legacy_C_ops as _C_ops
from .uniform import UniformObserver
from paddle.quantization.factory import ObserverFactory

class TokenQuantileObserver(ObserverFactory):
    r"""
    It collects maximum absolute values of target tensor under the condition of ignoring some tokens.

    Args:
        quant_bits (int): The number of bits for quantization.
        percentile(float): The percentage of bins that are retained when clipping the outliers.

    Examples:
       .. code-block:: python

            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import HistObserver
            quanter = HistObserver()
            q_config = QuantConfig(activation=quanter, weight=quanter)

    """

    def __init__(self, quant_bits=8, percentile=1.0):
        super(TokenQuantileObserver, self).__init__(quant_bits=quant_bits, percentile=percentile)

    def _get_class(self):
        return TokenQuantileObserverLayer


class TokenQuantileObserverLayer(UniformObserver):
    """token quantile
    """
    def __init__(
            self,
            layer,
            quant_bits=8, 
            percentile=1.0,):
        super(TokenQuantileObserverLayer, self).__init__(quant_bits=quant_bits)
        self._quant_bits = quant_bits
        self.cnt = 0
        self.percentile = 1.0
        self._min = None
        self._max = None

    def forward(self, inputs):
        """ Calculate forward pass.
        """
        if self.observer_enabled:
            x = inputs.clone().detach().reshape((-1, inputs.shape[-1])).cast('float32')
            x, lower, upper = self.prune_token(x)
            min_val_cur = lower#paddle.min(x)
            max_val_cur = upper#paddle.max(x)
            if self._max is None:
                self._min = min_val_cur
                self._max = max_val_cur
            else:
                self._min = self._min * self.cnt + min_val_cur
                self._max = self._max * self.cnt + max_val_cur
            self.cnt += 1
            self._min /= self.cnt
            self._max /= self.cnt

        if self.fake_quant:
            self.cal_thresholds()
            quant_inputs =  _C_ops.quantize_linear(
                inputs.cast('float32'),
                self._scale.cast('float32'),
                paddle.to_tensor(self._zero_point),
                "quant_axis",
                self.quant_axis(),
                "bit_length",
                self._quant_bits,
            ).cast(inputs.dtype)
            return _C_ops.dequantize_linear(
                quant_inputs.cast('float32'),
                self._scale.cast('float32'),
                paddle.to_tensor(self._zero_point),
                "quant_axis",
                self.quant_axis(),
                "bit_length",
                self._quant_bits,
            ).cast(inputs.dtype)

        return inputs

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def set_percentile(self, percentile):
        """Set percentile
        """
        self.percentile = percentile
    
    def quantile_range(self, tensor, dim=None):
        """  
        Compute the approximate quantile of a Paddle tensor.  
          
        Args:  
        - tensor (Tensor): Input tensor.  
        - dim (int, optional): Dimension along which to compute the quantile. If None, the tensor is flattened.  
          
        Returns:  
        - Tensor: Approximate quantile value.  
        """ 
        tensor = tensor.abs()
        if dim is not None:  
            # Keep the specified dimension and flatten the others  
            tensor = tensor.transpose((dim,) + tuple(i for i in range(tensor.dim()) if i != dim))  
            tensor = paddle.flatten(tensor, start_axis=1)  
        else:  
            tensor = paddle.flatten(tensor)  
              
        # Compute the index for the quantile  
        index_float = self.percentile * paddle.numel(tensor)  
        index = paddle.floor(index_float).astype('int64') - 1 
        # Sort the tensor in ascending order  
        sorted_tensor = paddle.sort(tensor)  
          
        # Get the quantile value using linear interpolation  
        upper = sorted_tensor[index] 
        if paddle.numel(tensor) > index + 1:
            upper += (index_float - index) * (sorted_tensor[index + 1] - sorted_tensor[index])  
          
        return -upper, upper

    def cac_thres(self, token_min, token_max):
        """ cal thres
        """
        _, upper = self.quantile_range(token_max)
        lower, _ = self.quantile_range(token_min)
        return lower, upper

    def prune_token(self, value):
        """prune token
        """
        token_max = value.max(1)
        token_min = value.min(1)
        indice_lower, indice_upper = self.cac_thres(token_min, token_max)
        value = paddle.clip(value, max=indice_upper, min=indice_lower)
        return value, indice_lower, indice_upper

    def min_value(self) -> float:
        return self._min

    def max_value(self) -> float:
        return self._max

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return -1

    def scales(self):
        """ Return output scales.
        """
        self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        self.cal_thresholds()
        return self._zero_point
