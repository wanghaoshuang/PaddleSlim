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

from typing import Union
import paddle
from paddle.quantization.config import QuantConfig
from paddle.quantization.factory import QuanterFactory


class SlimQuantConfig(QuantConfig):
    def __init__(self, activation: QuanterFactory, weight: QuanterFactory):
        super(SlimQuantConfig, self).__init__(activation, weight)
        self._constraints = []

    @property
    def constraints(self):
        return self._constraints

    def add_constraints(self, constraints):
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]
        self._constraints.extend(constraints)

    def add_qat_layer_mapping(self, source: Union[type, paddle.nn.Layer], target: type):
        r"""
        Add rules converting layers to simulated quantization layers
        before quantization-aware training. It will convert layers
        with type `source` to layers with type `target`. `source` and
        `target` should be subclass of `paddle.nn.Layer`. And a default
        mapping is provided by property `default_qat_layer_mapping`.

        Args:
            source(type): The type of layers that will be converted.
            target(type): The type of layers that will be converted to.

        Examples:
        .. code-block:: python

            from paddle.nn import Conv2D
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            q_config = QuantConfig(activation=None, weight=None)
            class CustomizedQuantedConv2D:
                def forward(self, x):
                    pass
                    # add some code for quantization simulation
            q_config.add_qat_layer_mapping(Conv2D, CustomizedQuantedConv2D)
        """
        assert ((isinstance(source, type) and issubclass(
            source, paddle.nn.Layer
        )) or isinstance(source, paddle.nn.Layer)), "The source layer to be placed should be a subclass of paddle.nn.Layer or instance of paddle.nn.Layer"
        assert isinstance(target, type) and issubclass(
            source, paddle.nn.Layer
        ), "The target layer should be a subclass of paddle.nn.qat.Layer"
        self._qat_layer_mapping[source] = target
        self._customized_qat_layer_mapping[source] = target

    def _get_qat_layer(self, layer: paddle.nn.Layer):
        q_config = self._get_config_by_layer(layer)

        target_type = self._customized_qat_layer_mapping.get(
            type(layer), self.qat_layer_mappings.get(type(layer))
        )
        target_type = self._customized_qat_layer_mapping.get(layer, target_type)
        return target_type(layer, q_config)