# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
Adaptive Bagging Quantization.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from paddleslim.quant.observers.asym_cachekv import AsymCacheKVObserverLayer
from paddleslim.utils.log import logger
from paddleslim.quant.observers import (
    AbsMaxChannelWiseWeightObserver,
    AbsmaxObserver,
    # AbsmaxTokenwiseObserver,
    AsymCacheKVObserver,
    AvgHeadwiseObserver,
    GroupWiseWeightObserver,
    KCacheChannelWiseObserver,
    TokenQuantileObserver,
)

from paddleslim.quant.layers.custom_attention import QuantizedCustomAttentionLayer

__all__ = ['AdaptiveBaggingQuant', 'QuantPolicy']


@dataclass
class QuantPolicy:
    """存储量化策略信息的数据类"""
    k_max: Optional[float] = None
    k_min: Optional[float] = None
    v_max: Optional[float] = None
    v_min: Optional[float] = None
    kv_loss: float = 0.0
    k_int4: int = 1
    v_int4: int = 1

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return asdict(self)

    def update_from_ratio(self, quant_info: dict, ratio: float):
        """根据 ratio 插值更新 k/v 的 max/min 值"""
        self.k_max = quant_info['k_max'] * ratio + quant_info['k_max_global'] * (1 - ratio)
        self.k_min = quant_info['k_min'] * ratio + quant_info['k_min_global'] * (1 - ratio)
        self.v_max = quant_info['v_max'] * ratio + quant_info['v_max_global'] * (1 - ratio)
        self.v_min = quant_info['v_min'] * ratio + quant_info['v_min_global'] * (1 - ratio)

class AdaptiveBaggingQuant:
    """
    Adaptive Bagging Quantization.
    """

    def __init__(self, args, model, search_iter):
        """
        Initializes the AdaptiveBaggingQuant class.

        Args:
            args: Arguments parsed from the command line.
            model: The neural network model.
            search_iter (int): The total iteration of the search process.
                Used to calculate average quantization loss.
        """
        self.quant_info = {}
        self.best_quant_policies = {}
        self.search_iter = search_iter
        self._init_quant_info(model)
        self.model = model

    def _init_quant_info(self, model):
        """
        Initializes the quantization information for each layer,
        including minimum value, maximum value, global minimum value, and global maximum value.
        """
        for cur_name, cur_layer in model.named_sublayers():
            print(cur_name, cur_layer)
        for cur_name, cur_layer in model.named_sublayers():
            if type(cur_layer) == AvgHeadwiseObserver:
                if '_observer' not in cur_name:
                    try:
                        layer_id = int(cur_name.split('.')[3])
                        kv = cur_name.split('.')[6][-1]
                    except:
                        layer_id = int(cur_layer.full_name().split("_")[5]) // 2
                        kv = cur_name.split('.')[2][-1]
                    if self.quant_info.get(layer_id) is None:
                        self.quant_info[layer_id] = {}
                    self.quant_info[layer_id][f"{kv}_min"] = cur_layer._min
                    self.quant_info[layer_id][f"{kv}_max"] = cur_layer._max
                    self.quant_info[layer_id][f"{kv}_min_global"] = cur_layer._min_global
                    self.quant_info[layer_id][f"{kv}_max_global"] = cur_layer._max_global
                    self.quant_info[layer_id]["search_iter"] = self.search_iter

    def _search_best_policy(self, kv_losses, layer_id, loss_kv_threshold=0.02, quant_bits=4):
        """
        search best quantizaion policy for each kv layer
        """
        kv_name = f"kv_layer_{layer_id}"
        quant_info = self.quant_info[layer_id]
        best_policy = QuantPolicy(kv_loss=loss_kv_threshold)
        best_ratio = None

        for ratio, loss_kv_cur in kv_losses[kv_name].items():
            if loss_kv_cur < best_policy.kv_loss:
                best_policy.kv_loss = loss_kv_cur
                best_policy.update_from_ratio(quant_info, ratio)
                best_ratio = ratio

        logger.debug(f"kv_name: {kv_name}, best_option_calib_kv: {best_ratio}")
        return best_policy.to_dict()

    def search(self):
        """search best quantizaion policy"""
        for cur_name, cur_layer in self.model.named_sublayers():
            if type(cur_layer) == QuantizedCustomAttentionLayer:
                kv_losses = cur_layer.kv_losses
                layer_id = cur_layer.layer_id
                self.best_quant_policies[int(layer_id)] = self._search_best_policy(
                    kv_losses, layer_id, loss_kv_threshold=10000
                )
                cur_layer.enable_fake_quant = False