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

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Any


@dataclass
class WeightActivationConfig:
    """Weight 和 Activation 量化配置"""
    weight_observer: Optional[str]  # Observer 类名
    weight_bits: Optional[int]
    weight_group_size: Optional[int]  # 仅用于 GroupWise
    activation_observer: Optional[str]
    activation_bits: Optional[int]


@dataclass
class CacheKVConfig:
    """CacheKV 量化配置"""
    enabled: bool
    bits: Optional[int]
    observer: Optional[str]


@dataclass
class QuantTypeConfig:
    """完整的量化配置"""
    weight_activation: WeightActivationConfig
    cachekv: CacheKVConfig


class WeightActivationType(Enum):
    """Weight 和 Activation 量化类型枚举"""
    
    # W8A8: weight=AbsMaxChannelWiseWeightObserver(8), activation=AbsmaxObserver(8)
    W8A8 = WeightActivationConfig(
        weight_observer="AbsMaxChannelWiseWeightObserver",
        weight_bits=8,
        weight_group_size=None,
        activation_observer="AbsmaxObserver",
        activation_bits=8
    )
    
    # WINT4/W4A16: weight=GroupWiseWeightObserver(4), activation=None
    WINT4 = WeightActivationConfig(
        weight_observer="GroupWiseWeightObserver",
        weight_bits=4,
        weight_group_size=128,  # 默认 group_size，可通过参数覆盖
        activation_observer=None,
        activation_bits=None
    )
    W4A16 = WINT4  # W4A16 与 WINT4 等价
    
    # WINT8/W8A16: weight=AbsMaxChannelWiseWeightObserver(8), activation=None
    WINT8 = WeightActivationConfig(
        weight_observer="AbsMaxChannelWiseWeightObserver",
        weight_bits=8,
        weight_group_size=None,
        activation_observer=None,
        activation_bits=None
    )
    W8A16 = WINT8  # W8A16 与 WINT8 等价
    
    # W4A8: weight=AbsMaxChannelWiseWeightObserver(4), activation=AbsmaxObserver(8)
    W4A8 = WeightActivationConfig(
        weight_observer="AbsMaxChannelWiseWeightObserver",
        weight_bits=4,
        weight_group_size=None,
        activation_observer="AbsmaxObserver",
        activation_bits=8
    )


class CacheKVType(Enum):
    """CacheKV 量化类型枚举"""
    
    # C8: AvgHeadwiseObserver(8, do_fp8_quant=True)
    C8 = CacheKVConfig(
        enabled=True,
        bits=8,
        observer="AvgHeadwiseObserver"
    )
    
    # C4: KCacheChannelWiseObserver(4) 或 AsymCacheKVObserver(4)
    C4 = CacheKVConfig(
        enabled=True,
        bits=4,
        observer="KCacheChannelWiseObserver"  # 默认，abq 模式下使用 AsymCacheKVObserver
    )
    
    # C2: 暂不支持
    C2 = CacheKVConfig(
        enabled=True,
        bits=2,
        observer=None  # 2bit 暂不支持
    )
    
    # C16 或无: 不启用 CacheKV 量化
    C16 = CacheKVConfig(
        enabled=False,
        bits=16,
        observer=None
    )
    NONE = C16  # 无后缀等价于 C16


class QuantType(Enum):
    """
    完整量化类型枚举，格式: {WeightActivationType}{CacheKVType}
    
    命名规则:
    - W{x}A{y}: Weight x-bit, Activation y-bit (A16 表示不量化 Activation)
    - WINT{x}: Weight-only 量化 x-bit
    - C{z}: CacheKV z-bit 量化 (C16 表示不量化 CacheKV)
    
    示例:
    - W8A8C8: Weight 8-bit, Activation 8-bit, CacheKV 8-bit
    - W4A16C4: Weight 4-bit, Activation 不量化, CacheKV 4-bit
    - WINT4: Weight 4-bit only, 无 Activation 和 CacheKV 量化
    """
    
    # ==================== W8A8 系列 ====================
    W8A8C8 = QuantTypeConfig(
        weight_activation=WeightActivationType.W8A8.value,
        cachekv=CacheKVType.C8.value
    )
    W8A8C4 = QuantTypeConfig(
        weight_activation=WeightActivationType.W8A8.value,
        cachekv=CacheKVType.C4.value
    )
    W8A8C16 = QuantTypeConfig(
        weight_activation=WeightActivationType.W8A8.value,
        cachekv=CacheKVType.C16.value
    )
    W8A8 = W8A8C16  # 无后缀等价于 C16
    
    # ==================== W4A8 系列 ====================
    W4A8C8 = QuantTypeConfig(
        weight_activation=WeightActivationType.W4A8.value,
        cachekv=CacheKVType.C8.value
    )
    W4A8C4 = QuantTypeConfig(
        weight_activation=WeightActivationType.W4A8.value,
        cachekv=CacheKVType.C4.value
    )
    W4A8C16 = QuantTypeConfig(
        weight_activation=WeightActivationType.W4A8.value,
        cachekv=CacheKVType.C16.value
    )
    W4A8 = W4A8C16  # 无后缀等价于 C16
    
    # ==================== WINT4/W4A16 系列 ====================
    WINT4C8 = QuantTypeConfig(
        weight_activation=WeightActivationType.WINT4.value,
        cachekv=CacheKVType.C8.value
    )
    WINT4C4 = QuantTypeConfig(
        weight_activation=WeightActivationType.WINT4.value,
        cachekv=CacheKVType.C4.value
    )
    WINT4C16 = QuantTypeConfig(
        weight_activation=WeightActivationType.WINT4.value,
        cachekv=CacheKVType.C16.value
    )
    WINT4 = WINT4C16  # 无后缀等价于 C16
    
    W4A16C8 = WINT4C8
    W4A16C4 = WINT4C4
    W4A16C16 = WINT4C16
    W4A16 = WINT4
    
    # ==================== WINT8/W8A16 系列 ====================
    WINT8C8 = QuantTypeConfig(
        weight_activation=WeightActivationType.WINT8.value,
        cachekv=CacheKVType.C8.value
    )
    WINT8C4 = QuantTypeConfig(
        weight_activation=WeightActivationType.WINT8.value,
        cachekv=CacheKVType.C4.value
    )
    WINT8C16 = QuantTypeConfig(
        weight_activation=WeightActivationType.WINT8.value,
        cachekv=CacheKVType.C16.value
    )
    WINT8 = WINT8C16  # 无后缀等价于 C16
    
    W8A16C8 = WINT8C8
    W8A16C4 = WINT8C4
    W8A16C16 = WINT8C16
    W8A16 = WINT8
    
    @classmethod
    def from_string(cls, quant_type_str: str) -> "QuantType":
        """
        从字符串解析量化类型
        
        Args:
            quant_type_str: 量化类型字符串，如 "W8A8C8", "WINT4", "W4A16C4"
            
        Returns:
            QuantType: 对应的枚举值
            
        Raises:
            ValueError: 如果量化类型字符串无效
        """
        try:
            return cls[quant_type_str.upper()]
        except KeyError:
            valid_types = [t.name for t in cls]
            raise ValueError(
                f"无效的量化类型: '{quant_type_str}'。"
                f"支持的量化类型: {valid_types}"
            )
    
    @property
    def weight_observer_name(self) -> Optional[str]:
        """获取 Weight Observer 类名"""
        return self.value.weight_activation.weight_observer

    @property
    def weight_observer(self) -> Optional[Type]:
        """获取 Weight Observer 类"""
        return get_observer_class(self.weight_observer_name)
    
    @property
    def weight_bits(self) -> Optional[int]:
        """获取 Weight 量化位数"""
        return self.value.weight_activation.weight_bits
    
    @property
    def activation_observer_name(self) -> Optional[str]:
        """获取 Activation Observer 类名"""
        return self.value.weight_activation.activation_observer
    
    @property
    def activation_observer(self) -> Optional[Type]:
        """获取 Activation Observer 类"""
        return get_observer_class(self.activation_observer_name)

    @property
    def activation_bits(self) -> Optional[int]:
        """获取 Activation 量化位数"""
        return self.value.weight_activation.activation_bits
    
    @property
    def cachekv_enabled(self) -> bool:
        """是否启用 CacheKV 量化"""
        return self.value.cachekv.enabled
    
    @property
    def cachekv_bits(self) -> Optional[int]:
        """获取 CacheKV 量化位数"""
        return self.value.cachekv.bits if self.value.cachekv.enabled else None
    
    @property
    def cachekv_observer_name(self) -> Optional[str]:
        """获取 CacheKV Observer 类名"""
        return self.value.cachekv.observer if self.value.cachekv.enabled else None

    @property
    def cachekv_observer(self) -> Optional[Type]:
        """获取 CacheKV Observer 类"""
        return get_observer_class(self.cachekv_observer_name)

# Observer 类名到实际类的映射表（用于动态导入）
OBSERVER_CLASS_MAP = {
    # Weight Observers
    "AbsMaxChannelWiseWeightObserver": "paddleslim.quant.observers.abs_max_weight.AbsMaxChannelWiseWeightObserverLayer",
    "GroupWiseWeightObserver": "paddleslim.quant.observers.groupwise.GroupWiseWeightObserverLayer",
    
    # Activation Observers
    "AbsmaxObserver": "paddleslim.quant.observers.abs_max.AbsmaxObserverLayer",
    "AbsmaxTokenwiseObserver": "paddleslim.quant.observers.abs_max_tokenwise.AbsmaxTokenwiseObserverLayer",
    "TokenQuantileObserver": "paddleslim.quant.observers.token_quantile.TokenQuantileObserverLayer",
    
    # CacheKV Observers
    "AvgHeadwiseObserver": "paddleslim.quant.observers.avg_headwise.AvgHeadwiseObserverLayer",
    "KCacheChannelWiseObserver": "paddleslim.quant.observers.kcache_channelwise.KCacheChannelWiseObserverLayer",
    "AsymCacheKVObserver": "paddleslim.quant.observers.asym_cachekv.AsymCacheKVObserverLayer",
}


def get_observer_class(observer_name: str) -> Type:
    """
    根据 Observer 名称获取对应的类
    
    Args:
        observer_name: Observer 类名
        
    Returns:
        Observer 类
        
    Raises:
        ValueError: 如果 Observer 名称无效
    """
    if observer_name not in OBSERVER_CLASS_MAP:
        raise ValueError(f"未知的 Observer: '{observer_name}'")
    
    module_path = OBSERVER_CLASS_MAP[observer_name]
    module_name, class_name = module_path.rsplit(".", 1)
    
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

