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
class RMSNorm(paddle.nn.Layer):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        """
            Initializes the instance of the class.
        
        Args:
            mean_dim (int): The dimension of the mean vector.
            eps (float, optional): The small value added for numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = paddle.create_parameter(shape=[mean_dim], dtype=paddle.get_default_dtype(), 
                          default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, x):
        """
            前向函数，计算输入的归一化结果。
        Args:
            x (Tensor): 输入张量，shape为（N, C）或者（N, C, H, W）。
            其中N是batch size，C是通道数，H和W是高和宽（可选）。
            输入张量的数据类型应为float16、float32或float64。
        Returns:
            Tensor: 返回一个与输入张量形状相同的张量，除了最后一维外，其他维度保持不变。
                    每个元素都是x的标准化值，使用方差和指定的epsilon进行修正。
            返回张量的数据类型与输入张量相同。
        """
        input_dtype = x.dtype
        x = x.cast('float32')
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * paddle.rsqrt(variance + self.eps)
        return self.weight * (x.cast(input_dtype))
