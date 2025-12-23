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

from paddle.quantization import QuantConfig
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleslim.quant.layers import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
    QuantizedCustomAttentionLayer
)
from paddleslim.common.wrapper_function import FuncWrapper
from paddleslim.quant.observers import (
    AbsMaxChannelWiseWeightObserver,
    AbsmaxObserver,
    AvgHeadwiseObserver,
    GroupWiseWeightObserver,
    KCacheChannelWiseObserver,
    AsymCacheKVObserver
)

from .quant_type import QuantType


def prepare_qconfig(args):
    """
    Prepare qconfig
    
    Args:
        args: 包含量化配置的参数对象，需要包含以下属性:
            - quant_type: 量化类型字符串，如 "W8A8C8", "WINT4" 等
            - group_size: (可选) GroupWise 量化的 group size
            - abq: (可选) 是否使用 ABQ 模式
    
    Returns:
        tuple: (activation, weight, cachekv, q_config)
            - activation: Activation Observer 实例或 None
            - weight: Weight Observer 实例或 None  
            - cachekv: CacheKV Observer 列表或 None
            - q_config: QuantConfig 实例
    """
    # 解析量化类型
    qt = QuantType.from_string(args.quant_type)
    
    # 创建 Weight Observer
    weight_bits = qt.weight_bits
    if qt.weight_observer_name == "GroupWiseWeightObserver":
        weight = GroupWiseWeightObserver(quant_bits=weight_bits, group_size=args.group_size)
    elif qt.weight_observer_name == "AbsMaxChannelWiseWeightObserver":
        weight = AbsMaxChannelWiseWeightObserver(quant_bits=weight_bits)
    else:
        weight = None
    
    # 创建 Activation Observer
    activation = None
    if qt.activation_observer_name == "AbsmaxObserver":
        activation = AbsmaxObserver(quant_bits=qt.activation_bits)
    
    # 配置 QuantConfig
    q_config = QuantConfig(activation=None, weight=None)
    q_config.add_qat_layer_mapping(ColumnParallelLinear, QuantizedColumnParallelLinear)
    q_config.add_qat_layer_mapping(RowParallelLinear, QuantizedRowParallelLinear)

    # 创建 CacheKV Observer
    cachekv = None
    if qt.cachekv_enabled:
        cachekv_bits = qt.cachekv_bits
        if cachekv_bits == 8:
            cachekv = [
                AvgHeadwiseObserver(quant_bits=cachekv_bits, moving_avg=True, quant_axis=1, do_fp8_quant=True),
                AvgHeadwiseObserver(quant_bits=cachekv_bits, moving_avg=True, quant_axis=1, do_fp8_quant=True)
            ]
        elif cachekv_bits == 4:
            if getattr(args, 'abq', False):
                cachekv = [
                    AsymCacheKVObserver(quant_bits=cachekv_bits, symmetric=False, quant_axis=[1, 3]),
                    AsymCacheKVObserver(quant_bits=cachekv_bits, symmetric=False, quant_axis=[1, 3])
                ]
            else:
                cachekv = [
                    KCacheChannelWiseObserver(quant_bits=cachekv_bits, symmetric=False),
                    KCacheChannelWiseObserver(quant_bits=cachekv_bits, symmetric=False)
                ]
        else:
            raise ValueError('cachekv_quant_bits should be 8 or 4, 2bit is not supported for now.')
        
        q_config.add_qat_layer_mapping(FuncWrapper, QuantizedCustomAttentionLayer)
    
    return activation, weight, cachekv, q_config

