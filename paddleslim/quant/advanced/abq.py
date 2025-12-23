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

__all__ = ['AdaptiveBaggingQuant']

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
        print(stop)
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
        best_k_scale = None
        best_k_min = None
        best_k_zp = None
        best_v_scale = None
        best_v_min = None
        best_v_zp = None

        best_option_calib_k = None
        best_option_calib_v = None
        best_quant_policy = {}
        kv_name = "kv_layer_" + str(layer_id)
        bnt = (1 << (quant_bits - 1)) - 1
        qmin = -bnt - 1
        qmax = bnt
        for ratio in kv_losses[kv_name].keys():
            loss_kv_cur = kv_losses[kv_name][ratio]
            k_max = self.quant_info[layer_id].get('k_max')
            k_min = self.quant_info[layer_id].get('k_min')
            k_max_global = self.quant_info[layer_id].get('k_max_global')
            k_min_global = self.quant_info[layer_id].get('k_min_global')
            v_max = self.quant_info[layer_id].get('v_max')
            v_min = self.quant_info[layer_id].get('v_min')
            v_max_global = self.quant_info[layer_id].get('v_max_global')
            v_min_global = self.quant_info[layer_id].get('v_min_global')

            k_max_cur = k_max * ratio + k_max_global * (1 - ratio)
            k_min_cur = k_min * ratio + k_min_global * (1 - ratio)
            v_max_cur = v_max * ratio + v_max_global * (1 - ratio)
            v_min_cur = v_min * ratio + v_min_global * (1 - ratio)

            if loss_kv_cur < loss_kv_threshold:
                loss_kv_threshold = loss_kv_cur
                best_option_calib_kv = ratio

                best_k_max = k_max_cur
                best_k_min = k_min_cur

                best_v_max = v_max_cur
                best_v_min = v_min_cur

        logger.debug(f"kv_name: {kv_name}, best_option_calib_kv: {best_option_calib_kv}")

        best_quant_policy["k_max"] = best_k_max
        best_quant_policy["k_min"] = best_k_min
        best_quant_policy["v_max"] = best_v_max
        best_quant_policy["v_min"] = best_v_min
        best_quant_policy["kv_loss"] = loss_kv_threshold
        best_quant_policy["k_int4"] = 1
        best_quant_policy["v_int4"] = 1

        return best_quant_policy

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