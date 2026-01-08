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

import numpy as np
import paddle
import paddle.nn as nn
from paddle.quantization.base_observer import BaseObserver
from .utils import get_ln_linear_info, find_parent_layer_and_sub_name
from .utils_layers import ShiftSmoothHelpLayer, WOBiasHelpLayer
from ...utils.log import logger
__all__ = ['TokenWiseClipping']


class TokenWiseClipping():
    """
    Args:
        model(paddle.nn.Layer, required): the model to be smoothed 
        model_config (dict, required): the config of model to be smoothed 
        step(float, default=0.002): the step size used during clipping process
        iters(int, default=20): the number of iterations used during clipping process
        
    Examples:
    .. code-block:: python
   
    from paddleslim.quant.advanced import Smooth
    token_clip = TokenWiseClipping(model)
    fp_input = []
    fp_output = []
    for data in dataloader():
        fp_input.append(data)
        out = model(data)
        fp_output.append(out)
    token_clip.token_wise_clipping(fp_input, fp_output)
    """
    def __init__(
            self,
            model,
            model_config=None,
            step=0.002,
            iters=20):

        self.model = model
        self.model_config = model_config
        
        self.model.eval()
        self.step = step
        self.iters = iters
    
    def enable_quantization(self):
        """
            启用量化，禁用observer并启用伪量化。
        
        Args:
            None.
        
        Returns:
            None.
        """
        for name, submodule in self.model.named_sublayers():
            if isinstance(submodule, BaseObserver):
                submodule.disable_observer()
                submodule.enable_fake_quant()

    def calibrate(self, fp_input, fp_output=None, cal_loss=False):
        """
            对模型进行校准，可选择计算损失值。
        如果cal_loss为True，则会返回一个浮点数，表示模型在给定输入和标签的情况下的平均损失值；否则，不会返回任何值。
        
        Args:
            fp_input (Iterator[Dict[str, Tensor]]): 包含输入数据的迭代器，每个元素是一个字典，其中包含了所需的输入张量。
            fp_output (Optional[List[Tensor]], optional): 包含标签数据的列表，默认为None。如果cal_loss为True，则必须提供此参数。
            cal_loss (bool, optional): 是否计算损失值，默认为False。
        
        Returns:
            Optional[float]: 如果cal_loss为True，则返回一个浮点数，表示模型在给定输入和标签的情况下的平均损失值；否则，不会返回任何值。
        """
        loss = 0
        loss_func = nn.MSELoss()  
        for i, batch in enumerate(fp_input):
            if cal_loss:
                output = self.model(**batch)[0]
                loss += loss_func(output, fp_output[i]) 
            else:
                self.model(**batch)
        return loss

    def set_ratio(self, ratio):
        """
            设置比例，用于TokenQuantileObserverLayer。
        参数：
            ratio (float, required): 比例，取值范围为[0,1]。
        返回值：
            None，无返回值。
        """
        from paddleslim.quant.observers.token_quantile import TokenQuantileObserverLayer 
        for cur_name, module in self.model.named_sublayers():
            if isinstance(module, BaseObserver):
                module.disable_fake_quant()
                module.enable_observer()
                if isinstance(module, TokenQuantileObserverLayer):
                    module.set_percentile(ratio)
                    module.cnt = 0
    
    def find_ratio(self, fp_input, fp_output):
        """
            搜索ratio，通过对比输出的精度来确定最佳ratio。
        参数：
            fp_input (list): 模型输入。
            fp_output (list, optional): 全精度模型输出（默认为None）。
        返回值：
            None，但会在控制台打印出ratio。
        """
        p, loss = 0, None
        for i in range(self.iters):
            self.set_ratio(1.0 - self.step * i)
            self.calibrate(fp_input)
            self.enable_quantization()
            cur_loss = self.calibrate(fp_input, fp_output=fp_output, cal_loss=True)
            logger.info('the ratio is {}, the loss is {}'.format(1.0 - self.step * i, float(cur_loss.cast('float32'))))
            if loss is None or loss >= cur_loss:
                loss = cur_loss
                p = i
        ratio = 1.0 - self.step * p
        logger.info('the best percentile is {}'.format(ratio))
        self.set_ratio(ratio)

    def token_wise_clipping(self, fp_input, fp_output):
        """
            进行token wise clipping
        
        Args:
            fp_input (list): 模型输入。
            fp_output (list): 模型输出。
        
        Returns:
            None.
        """

        logger.info("*** Evaluate Token Percentile ***")
        self.find_ratio(fp_input, fp_output)
