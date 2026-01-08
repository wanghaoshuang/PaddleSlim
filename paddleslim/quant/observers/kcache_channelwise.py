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
"""
KCacheChannelWiseObserver.
"""
import numpy as np
import paddle
from .channel_wise import ChannelWiseObserver
from paddle.quantization.factory import ObserverFactory


class KCacheChannelWiseObserver(ObserverFactory):
    r"""
    It collects channel-wise maximum absolute values of target weights.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import KCacheChannelWiseObserver
            quanter = KCacheChannelWiseObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(self, quant_bits=8, quant_axis=-1, symmetric=False):
        super(KCacheChannelWiseObserver, self).__init__(
            quant_bits=quant_bits,
            quant_axis=quant_axis,
            symmetric=symmetric)

    def _get_class(self):
        return KCacheChannelWiseObserverLayer


class KCacheChannelWiseObserverLayer(ChannelWiseObserver):
    """
    KCacheChannelWiseObserverLayer
    """
    def __init__(self, layer, quant_bits=8, quant_axis=-1, symmetric=False):
        super(KCacheChannelWiseObserverLayer, self).__init__(
            layer,
            quant_bits=quant_bits,
            sign=True,
            symmetric=symmetric, 
            quant_axis=quant_axis)
        self.quant_bits = quant_bits
        self.calibration_loss = float('inf')
        self.qmin, self.qmax = self.qmin_qmax
        self._layer = layer
        self._max = None
        self._scale = None
        self._zero_point = None
        self._symmetric = symmetric
        

    def forward(self, inputs):
        """
        forward function of KCacheChannelWiseObserverLayer
        """
        if self._symmetric:
            self._max = self._cal_abs_max(inputs)
        else:
            self._min, self._max = self._cal_abs_max(inputs)
        return inputs

    def _cal_abs_max(self, inputs):
        """
        cal abs max value of inputs
        """
        # inputs: [bsz, num_heads, seq_len, head_dim]
        reduce_axis = [0, 2]
        if self._symmetric:
            # use keepdim when ruduce to avoid reshape in Paddle format.py
            abs_max_values = paddle.max(paddle.abs(inputs), axis=reduce_axis, keepdim=True).cast("float32")
            abs_max_values = paddle.where(abs_max_values == np.float32(0.0),
                                        np.float32(1e-8), abs_max_values)
            if self._max is not None:
                abs_max_values = paddle.maximum(abs_max_values, self._max)
            return abs_max_values
        else:
            max_values = paddle.max(inputs, axis=reduce_axis, keepdim=True)
            min_values = paddle.min(inputs, axis=reduce_axis, keepdim=True)

            if self._max is not None:
                max_values = paddle.maximum(max_values, self._max)
                min_values = paddle.minimum(min_values, self._min)

            return min_values, max_values


    def min_value(self) -> float:
        """
        Return min value.
        """
        if self._symmetric:
            return 0.
        return self._min

    def max_value(self) -> float:
        """
        Return max value.
        """
        return self._max

    def cal_thresholds(self):
        """ 
        Compute thresholds for MAX function.
        """
        if self._symmetric:
            self._scale = self._max
            self._zero_point = paddle.zeros_like(self._scale)
        else:
            self._scale, self._zero_point = self.cal_scales_zero_points()
        

    def scales(self):
        """ 
        Return output scales.
        """
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """ 
        Return output zero points.
        """
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point

    