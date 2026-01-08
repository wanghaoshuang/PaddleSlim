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

import paddle
import numpy as np
import paddle.distributed.fleet as fleet
from .utils import compute_scales, k_means
from .metrics import mse_loss
from ...utils.log import logger

__all__ = ['SmoothSearchV2']

class SmoothSearchV2():
    """
    NewSearch provides to search k_piece, alpha and scale.

    Args:
    bits_length (int): Number of bits to quantize the weight. Default: 8.
    search_min (float): Minimum scale for search. Default: 0.1.
    search_step (int): Step for search. Default: 100.
    weight_quant_method (str): Weight quantization method. Choosen from abs_max, abs_max_channel_wise and avg. 
                               Default: abs_max_channel_wise.
    act_quant_method (str): Activation quantization method. Choosen from abs_max, avg. Default: abs_max.
    loss_function (callable): Loss function. Default: mse_loss.
    """
    def __init__(self,
                 weight_bits_length=4,
                 act_bits_length=8,
                 search_min=0.1,
                 search_step=100,
                 weight_quant_method='abs_max_channel_wise',
                 act_quant_method='abs_max',
                 loss_function=mse_loss,
                 dp_degree=1):
        self.weight_bits_length = weight_bits_length
        self.act_bits_length = act_bits_length
        self.search_min = search_min
        self.search_step = search_step
        self.weight_quant_method = weight_quant_method
        self.act_quant_method = act_quant_method
        self.weight_bnt = (1 << (self.weight_bits_length - 1)) - 1
        self.act_bnt = (1 << (self.act_bits_length - 1)) - 1
        self.loss_function = loss_function
        self.dp_degree = dp_degree
        try:
            hcg = fleet.get_hybrid_communicate_group()
            self.mp_degree = hcg.get_model_parallel_world_size()
            self.mp_id = hcg.get_model_parallel_rank()
            self.dp_id = hcg.get_data_parallel_rank()
        except:
            self.mp_degree = 1
            self.mp_id = 0
            self.dp_id = 0


    def search(self, layer_name, sampled_input, act_abs_max, weight):
        """
            搜索层的输入，以获得最佳的量化范围。
        
        Args:
            layer_name (str): 要搜索的层名称。
            sampled_input (Tensor): 采样的层输入。类型为Tensor，形状为（batch_size, ...）。
            act_abs_max (Tensor): 层输入的绝对值最大值。类型为Tensor，形状为（batch_size, ...）。
            weight (Tensor): 层的权重。类型为Tensor，形状为（output_channels, input_channels）。
        
        Returns:
            Tensor, shape (batch_size, ...): 返回最佳的量化范围。如果是分布式训练，则返回所有进程中的最佳量化范围。
        """
        act_abs_max_ = act_abs_max.detach().clone().cast('float32')
        act = sampled_input.detach().clone().cast('float32')
        weight_ = weight.detach().clone().cast('float32')
        act.stop_gradient = True
        logger.debug('[smooth search] search input of %s' % layer_name)
        dtype = weight_.dtype
        origin_out = paddle.matmul(act, weight_)
        w_abs_max = weight_.abs().max(axis=-1, keepdim=True)
        rw_abs_max = w_abs_max.reshape(act_abs_max_.shape)
    
        smooth_scale_out = None
        best_scale = None
    
        calibration_loss = float('inf')
        alpha_max = act_abs_max_.max() 
        alpha_min = min(paddle.to_tensor(self.search_min, dtype='float32'), 0.2 * alpha_max)
        step = float((alpha_max - alpha_min) / self.search_step)
        alpha = alpha_min
        final_alpha = None
        alpha_range = np.arange(alpha_min, alpha_max, step).tolist()
        if self.search_step % self.dp_degree != 0:
            additional_range = [alpha_max] * (((self.search_step // self.dp_degree) + 1) 
                   * self.dp_degree - self.search_step)
            alpha_range = alpha_range + additional_range
        
        for i in range(len(alpha_range) // self.dp_degree):
            alpha = alpha_range[self.dp_id + i * self.dp_degree]
            act_abs_max_tmp = act_abs_max_.detach().clone()
            s = paddle.where(act_abs_max_tmp > alpha, act_abs_max_tmp / alpha, paddle.to_tensor(1, dtype='float32'))
            del act_abs_max_tmp
            smooth_scale_tmp = s
            new_act = act / smooth_scale_tmp
            new_weight = weight_ * smooth_scale_tmp.reshape(
                w_abs_max.shape)
    
            quant_scale = compute_scales(
                new_act, method=self.act_quant_method)
            quant_act = paddle.clip(
                paddle.round(new_act / quant_scale * self.act_bnt),
                -self.act_bnt, self.act_bnt)
            quant_dequant_act = quant_act / self.act_bnt * quant_scale
    
            quant_scale_w = compute_scales(
                new_weight, method=self.weight_quant_method)
            quant_weight = paddle.clip(
                paddle.round(new_weight / quant_scale_w * self.weight_bnt),
                -self.weight_bnt, self.weight_bnt)
            quant_dequant_weight = quant_weight / self.weight_bnt * quant_scale_w
            
            new_out = paddle.matmul(quant_dequant_act,
                                    quant_dequant_weight)
    
            cur_loss = self.loss_function(origin_out, new_out)
            if cur_loss <= calibration_loss:
                calibration_loss = cur_loss
                final_smooth_scale = smooth_scale_tmp
                final_alpha = alpha
        
        logger.debug("Layer {}, loss: {}, alpha : {}".format(layer_name, float(calibration_loss), float(final_alpha)))
        if self.dp_degree > 1:
            smooth_scale_out_list = []
            calibration_loss_list = []

            paddle.distributed.all_gather(smooth_scale_out_list, final_smooth_scale)
            paddle.distributed.all_gather(calibration_loss_list, calibration_loss)

            for i in range(self.dp_degree):
                if calibration_loss_list[i * self.mp_degree + self.mp_id] < calibration_loss:
                    calibration_loss = calibration_loss_list[i * self.mp_degree + self.mp_id]
                    final_smooth_scale = smooth_scale_out_list[i * self.mp_degree + self.mp_id]

        if smooth_scale_out is None:
            smooth_scale_out = final_smooth_scale
        return smooth_scale_out
