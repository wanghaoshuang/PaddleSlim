# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


import functools
from paddle.nn import functional as F

from paddle.nn import Layer
from paddle.nn.quant.format import ConvertibleQuantedLayer

from paddle.quantization.base_observer import BaseObserver

from paddle.nn.quant.format import LinearQuanter, LinearDequanter
from paddle.distributed.fleet.utils import recompute


class QuantedRecomputeLinear(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedLinear is the same as Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()
        # For Linear
        self.weight = layer.weight
        self.bias = layer.bias
        self.name = layer.name
        # For FakeQuant

        self.weight_observer = None
        self.activation_observer = None
        if q_config.weight is not None:
            self.weight_observer = q_config.weight._instance(layer)
            assert isinstance(self.weight_observer, BaseObserver)
        if q_config.activation is not None:
            self.activation_observer = q_config.activation._instance(layer)
            assert isinstance(self.activation_observer, BaseObserver)

    def forward(self, input):
        quant_input = input
        quant_weight = self.weight

        if self.activation_observer is not None:
            self.activation_observer(quant_input)
            activation_quanter = LinearQuanter.from_quanter(self.activation_observer)
            activation_dquanter = LinearDequanter.from_quanter(self.activation_observer)
            quant_input = activation_quanter(quant_input)

        if self.weight_observer is not None:
            self.weight_observer(quant_weight)
            weight_quanter = LinearQuanter.from_quanter(self.weight_observer)
            weight_dquanter = LinearDequanter.from_quanter(self.weight_observer)
        
        linear_forward = functools.partial(self._recompute_linear_forward, act_dequanter=activation_dquanter, weight_quanter=weight_quanter, weight_dequanter=weight_dquanter)
        
        outputs = recompute(linear_forward, 
                            quant_input, # quantized activation with int8 dtype
                            quant_weight, # original weights with fp16 dtype
                            use_reentrant=self.layer.config.recompute_use_reentrant,
        )

        return outputs

    def _recompute_linear_forward(self, quantized_input,
                                weight,
                                act_dequanter=None,
                                weight_quanter=None,
                                weight_dequanter=None,):
        if act_dequanter is not None:
            quantized_input = act_dequanter(quantized_input)
        if weight_dequanter is not None and weight_quanter is not None:
            weight = weight_dequanter(weight_quanter(weight))

        self._linear_forward(quantized_input, weight)

    def _linear_forward(self, input, weight):
        out = F.linear(x=input, weight=weight, bias=self.bias, name=self.name)
        return out

    def weights_to_quanters(self):
        return [('weight', 'weight_observer')]

    def activation_quanters(self):
        return ['activation_observer']
