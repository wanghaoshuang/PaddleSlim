# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""
import math
import time
import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear

from ..observers.abs_max import AbsmaxObserverLayer

from .utils import compute_scales, fake_quant,compute_scales_moe
from ...utils.log import logger

__all__ = ['moe_shared_scale']


class moe_shared_scale(nn.Layer):
    """
    """
    def __init__(self,
                 model,
                 quant_bits=8,
                 quant_method='abs_max'):
        super(moe_shared_scale, self).__init__()
        self.model = model
        self.bnt = (1 << (quant_bits - 1)) - 1
        quant_bits = (1 << (quant_bits - 1)) - 1
        self.quant_bits = paddle.to_tensor(quant_bits, dtype='float32')
        self.quant_method = quant_method
        self.sampled_num = {}
        self.sampled_allinputs ={}
        self.tmpzerp=None
        self._apply_hook()
        self.iter=20
        self.sampled_concatinputs=[]

    def _apply_hook(self):
        self._forward_hook_list = []
        for _, sub_layer in self.model.named_sublayers():
            if type(sub_layer) in [AbsmaxObserverLayer]:
                # if 'expert' not in _:
                #     print('skip gptq')
                # else:
                #     print("GPTQ Quantizing Layer: ", _)
                if 'up_gate_proj' in _:
                    forward_pre_hook_handle = sub_layer.register_forward_pre_hook(
                        self._forward_pre_hook)
                    self._forward_hook_list.append(forward_pre_hook_handle)

    def _forward_pre_hook(self, layer, input):
        self._sample(input, layer.full_name())
        return input

    def _sample(self, input, layer_name):
        inp = input[0] if type(input) == tuple else input
        inp = inp.cast('float32')
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # if len(inp.shape) == 3:
        #     inp = inp.reshape((-1, inp.shape[-1]))
        # inp = inp
        if self.tmpzerp is None:
            self.tmpzerp=paddle.zeros([1,1,inp.shape[-1]],dtype='float32')
        if layer_name not in self.sampled_allinputs:
            self.sampled_allinputs[layer_name] = [inp]
            self.sampled_num[layer_name] = 0
        else:
            self.sampled_num[layer_name] += tmp
            self.sampled_allinputs[layer_name].append(inp)
        del inp

    def search_best_scale(self):
        """
        fasterquant
        """
        self.bestscales=[]
        scales=[]
        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.sampled_allinputs:
                fp_input=self.sampled_allinputs[layer_name]
                cnt=0
                fp_input=paddle.concat(fp_input,axis=1)
                fp_input=paddle.sort(fp_input, axis=- 1, descending=True)
                for i in range(fp_input.shape[1]):
                    import copy
                    temp=copy.deepcopy(fp_input[:,i,:])
                    is_all_zeros = paddle.all(temp == 0)
                    if is_all_zeros:
                        cnt=i
                        break
                fp_input=fp_input[:,:cnt,:]
                scales.append(sub_layer.scales())
                self.sampled_concatinputs.append(fp_input)
                self.sampled_num[layer_name]=cnt
        # print('scales')
        # print(scales)
        # print('fp input')
        # print(fp_input)
        # print(len(self.sampled_concatinputs))
        # print(self.sampled_concatinputs)
        all_expert_input=paddle.concat(self.sampled_concatinputs,axis=1)
        # print(self.sampled_allinputs)
        # print(self.sampled_num)
        scales=paddle.stack(scales)

        scales_max=paddle.max(scales).cast('float32')
        scales_mean=paddle.mean(scales,axis=-1).cast('float32')
        scales_std=paddle.std(scales,axis=-1).cast('float32')
        self.step = paddle.to_tensor(20.0).cast('float32')
        self.step=paddle.round(self.step*scales_std)
        self.iter=int(self.step.item())
        cur_loss = 0
        loss_func = nn.MSELoss() 
        best_scale = scales_max
        calibration_loss = float('inf')
        for i in range(self.iter):
            best_scale_tmp =  scales_max - paddle.to_tensor(i).cast('float32') *(scales_max-scales_mean)/paddle.to_tensor(self.iter).cast('float32')

            quant_act = paddle.clip(
                    paddle.round(all_expert_input / best_scale_tmp * self.bnt),
                    -self.bnt, self.bnt)
            quant_dequant_act = quant_act / self.bnt * best_scale_tmp
            cur_loss = loss_func(all_expert_input,quant_dequant_act)
            if cur_loss <= calibration_loss:
                calibration_loss = cur_loss
                best_scale = best_scale_tmp
        print("Best scale :", best_scale)
        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.sampled_allinputs:
                sub_layer._scale=best_scale

        


        # quant_act = paddle.clip(
        #         paddle.round(fp_input/ quant_scale * self.bnt),
        #         -self.bnt, self.bnt)

        # cur_loss = self.loss_function(quant_act , fp_input)
        #     if cur_loss <= calibration_loss:
        #         calibration_loss = cur_loss
        #         final_smooth_scale = smooth_scale_tmp
        #         final_alpha = alpha
        del self.sampled_num,self.sampled_allinputs,self.sampled_concatinputs
        self._remove_hook()
        paddle.device.cuda.empty_cache()

    def _remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = [] 