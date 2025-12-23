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

import abc
from typing import Tuple
import numpy as np
import paddle
from paddle.quantization.base_observer import BaseObserver


class UniformObserver(BaseObserver):
    """This is the base class for a uniform quantization observer, which provides
    common functions for calculating the scale and zero-point used in uniform quantization.
    Uniform quantization maps floating point values to integers, where the scale determines
    the step size of the quantizer and the floating point zero is mapped to the zero-point,
    an integer value ensuring that zero is quantized without error.

    Args:
        quant_bits (int): The number of bits for quantization.
        sign (bool): Whether the quantized integer includes a sign.
        symmetric (bool): Whether it is symmetric quantization. the quantization is symmetric.
        In symmetric quantization, the range of floating point values is relaxed to be symmetric
        around zero and the zero-point is always 0.

    """

    def __init__(
        self,
        quant_bits=8,
        sign=True,
        symmetric=True,
    ):
        super(UniformObserver, self).__init__()
        self._quant_bits = quant_bits
        self._sign = sign
        self._symmetric = symmetric

        self._min = None
        self._max = None
        self._qmin = None
        self._qmax = None

        self._scale = None
        self._zero_point = None

    @property
    def qmin_qmax(self):
        """Calculate the range of the quantized integer based on the specified
        quant_bits, sign, and symmetric properties."""
        if isinstance(self._quant_bits, tuple):
            if self._quant_bits[0] == 4 and self._quant_bits[1] == 3 and len(self._quant_bits) == 2:
                self._qmin = -448.0
                self._qmax = 448.0
            elif self._quant_bits[0] == 5 and self._quant_bits[1] == 2 and len(self._quant_bits) == 2:
                self._qmin = -57344.0
                self._qmax = 57344.0
            else:
                raise NotImplementedError(
                    "Currently, only float8_e4m3 and float8_e5m2 formats are supported. Please set quant_bits to (4,3) or (5,2) for the corresponding format."
                )
        else:
            if self._sign:
                self._qmin = -(2 ** (self.bit_length() - 1))
                self._qmax = 2 ** (self.bit_length() - 1) - 1
            else:
                self._qmin = 0
                self._qmax = 2 ** self.bit_length()
        return self._qmin, self._qmax

    @abc.abstractmethod
    def min_value(self) -> float:
        """ The minimum value of floating-point numbers."""
        raise NotImplementedError(
            "Please implement the abstract method to get the The minimum value of floating-point numbers."
        )

    @abc.abstractmethod
    def max_value(self) -> float:
        """ The maximum value of floating-point numbers."""
        raise NotImplementedError(
            "Please implement the abstract method to get the the maximum value value of floating-point numbers."
        )

    def cal_scales_zero_points(self) -> Tuple[float, float]:
        """ Calculate the scales and zero points based on the min_value and max_value.
        """
        assert self.min_value() is not None and self.max_value() is not None
        _qmin, _qmax = self.qmin_qmax
        # For one-sided distributions, the range (_min , _max ) is relaxed to include zero.
        # It is important to ensure that common operations like zero padding do not cause quantization errors.
        _min = min(self.min_value(), 0.)
        _max = max(self.max_value(), 0.)

        if self._symmetric:
            self._scale = max(-_min, _max)
            if self._sign:
                self._zero_point = 0
            else:
                self._zero_point = (_qmax + _qmin) / 2
        else:
            self._scale = (_max - _min) / float(_qmax - _qmin)
            self._zero_point = _qmin - round(_min / self._scale)
            self._zero_point = np.clip(self._zero_point, _qmin, _qmax)
        return self._scale, self._zero_point

    def gather_scale(self):
        """
        在分布式训练中，汇聚各进程的 scale 并返回全局最大 scale。

        分布式并行模式下 scale 的合并逻辑：
        -----------------------------------------------------------------------
        1. 纯数据并行 (DP only, dp_degree > 1, mp_degree = 1):
           - 每个 DP rank 独立计算 scale，数据分片不同导致 scale 可能不同
           - all_gather 收集所有 DP rank 的 scale
           - 取所有 scale 的最大值，确保量化范围覆盖所有数据分片

        2. 纯模型并行 (MP only, dp_degree = 1, mp_degree > 1):
           - 各 MP rank 持有模型的不同部分，scale 相互独立
           - 无需跨 rank 合并，直接返回当前 rank 的 scale

        3. 混合并行 (DP + MP, dp_degree > 1, mp_degree > 1):
           - all_gather 收集所有 rank 的 scale，总数为 dp_degree * mp_degree
           - 通过 mp_id 筛选出同一 MP group 内的所有 DP rank 的 scale
           - 对这些 scale 取最大值，保证同一模型分片在所有数据分片上的量化一致性
        -----------------------------------------------------------------------

        Returns:
            paddle.Tensor: 汇聚后的全局 scale
        """
        from paddle.distributed import fleet

        scale = self.scales()

        # 获取分布式并行参数
        hcg = fleet.get_hybrid_communicate_group()
        dp_degree = hcg.get_data_parallel_world_size()
        mp_degree = hcg.get_model_parallel_world_size()
        mp_id = hcg.get_model_parallel_rank()

        if dp_degree > 1:
            # 收集所有 rank 的 scale
            scale_list = []
            paddle.distributed.all_gather(scale_list, scale)

            # 筛选同一 MP group 内各 DP rank 的 scale 并堆叠
            # scale_list 的排列顺序: [dp0_mp0, dp0_mp1, ..., dp1_mp0, dp1_mp1, ...]
            # 索引公式: rank_idx = dp_rank * mp_degree + mp_id
            same_mp_group_scales = []
            for dp_rank in range(dp_degree):
                rank_idx = dp_rank * mp_degree + mp_id
                # 增加一个维度用于后续 concat
                expanded_scale = paddle.unsqueeze(scale_list[rank_idx], axis=0)
                same_mp_group_scales.append(expanded_scale)

            # 沿 axis=0 拼接后取最大值，得到全局 scale
            stacked_scales = paddle.concat(same_mp_group_scales, axis=0)
            gathered_scale = stacked_scales.max(axis=0, keepdim=False)

            # 更新内部 scale
            paddle.assign(gathered_scale, self._scale)
            return gathered_scale
        else:
            return scale

    