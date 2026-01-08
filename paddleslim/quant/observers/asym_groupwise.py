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
from .channel_wise import ChannelWiseObserver
from paddle.quantization.factory import ObserverFactory


class AsymGroupwiseObserver(ObserverFactory):
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
            from paddle.quantization.quanters import AbsMaxGroupwiseObserver
            quanter = AbsMaxGroupwiseObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(self, quant_bits=8, symmetric=True):
        super(AsymGroupwiseObserver, self).__init__(
            quant_bits=quant_bits, symmetric=symmetric)

    def _get_class(self):
        return AsymGroupwiseObserverLayer


class AsymGroupwiseObserverLayer(ChannelWiseObserver):
    def __init__(self, layer, quant_bits=8, symmetric=False):
        super(AsymGroupwiseObserverLayer, self).__init__(
            layer,
            quant_bits=quant_bits,
            sign=True,
            symmetric=symmetric, )
        self.quant_bits = quant_bits
        self.qmin, self.qmax = self.qmin_qmax
        self._layer = layer
        self._max = None
        self._min = None
        self._scale = None
        self._zero_point = None

    def forward(self, inputs):
        self._min, self._max = self._cal_abs_max(inputs)
        return inputs

    def _cal_abs_max(self, inputs):
        reduce_axis = tuple(
            [i for i in range(len(inputs.shape)) if i != self.quant_axis()])

        max_values = paddle.max(inputs.cast("float32"), axis=reduce_axis)
        min_values = paddle.min(inputs.cast("float32"), axis=reduce_axis)

        if self._max is not None:
            max_values = paddle.maximum(max_values, self._max)
            min_values = paddle.minimum(min_values, self._min)

        return min_values, max_values

    def min_value(self) -> float:
        return self._min

    def max_value(self) -> float:
        return self._max

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def scales(self):
        """ Return output scales.
        """
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point