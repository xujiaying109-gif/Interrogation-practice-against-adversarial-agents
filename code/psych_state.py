"""
心理状态机
"""
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass
import random


from typing import Dict, Optional
from dataclasses import dataclass
import random
from personality import BasePersonality


@dataclass
class InterrogationResult:
    round_num: int
    defense: float
    stress: float
    strategy: str
    defense_change: float
    stress_change: float


class MentalStateMachine:
    """
    基于非线性动态计算的心理状态机 (基于 Strategy Pattern 重构)
    """
    # 策略基础修正 (仍保留作为基础参考，具体伤害交由性格类裁定)
    STRATEGIES = {
        "心理施压": {"defense_effect": 1.3, "stress_effect": 1.6},
        "出示证据": {"defense_effect": 1.6, "stress_effect": 1.1},
        "连续追问": {"defense_effect": 1.0, "stress_effect": 1.2},
        "正常询问": {"defense_effect": 0.8, "stress_effect": 0.8},
        "缓和安抚": {"defense_effect": 0.2, "stress_effect": 0.1},
    }

    def __init__(self, personality: BasePersonality, total_rounds: int = 10):
        """
        初始化时必须传入 BasePersonality 实例。total_rounds 保留作接口兼容，不再用于核心步长计算。
        """
        self.personality = personality

        self.defense = personality.initial_defense
        self.stress = personality.initial_stress
        self.current_round = 0
        self.total_rounds = total_rounds

    def execute_round(self, strategy: str, evidence_strength: float = 0.5, is_trap: bool = False) -> InterrogationResult:
        self.current_round += 1

        # 获取基础策略系数
        strat_conf = self.STRATEGIES.get(strategy, self.STRATEGIES["正常询问"])
        
        # 1. 基础浮动：减缓节奏，基础变动在 1-4 之间动态计算
        base_def_drop = max(1.0, self.defense * 0.04) 
        base_str_rise = max(1.0, (100.0 - self.stress) * 0.05)

        # 2. 引入真实的外界刺激影响 (证据确凿度与诈供陷阱)
        # 真实铁证触发暴击伤害 (削弱暴击秒杀)
        if not is_trap and evidence_strength >= 0.8 and strategy == "出示证据":
            base_def_drop *= 1.5
            base_str_rise *= 1.5
            
        # 识破诈供触发防线反弹
        elif is_trap and self.defense > 30: # 还有理智的人能识破陷阱
            # 防御反弹，压力小幅回落
            base_def_drop = -5.0  # 负数表示防御增加
            base_str_rise = -3.0   # 负数表示压力减小

        # 结合基础步长、策略系数计算出 raw_drop
        raw_def_drop = base_def_drop * strat_conf["defense_effect"]
        raw_str_rise = base_str_rise * strat_conf["stress_effect"]

        # 3. [核心多态调用]：把 raw_drop 喂给性格对象，让其自行计算最终伤害
        def_change = self.personality.calculate_defense_damage(raw_def_drop, strategy)
        str_change = self.personality.calculate_stress_increase(raw_str_rise, strategy)

        # 修复 Bug：如果是反弹（负的 raw_drop），经过性格类的正数常数乘后可能符号没变，但也可能由于某些计算异常需要强制同号
        # 性格类的乘数都是正数，因此符号通常会保留，但安全起见：
        if raw_def_drop < 0 and def_change > 0:
            def_change = -def_change
        if raw_str_rise < 0 and str_change > 0:
            str_change = -str_change

        # 引入随机波动因子，让表现更真实
        # 对于负数计算边界需特殊处理
        vol = self.personality.volatility
        if def_change >= 0:
            def_change *= random.uniform(1.0 - vol, 1.0 + vol)
        else:
            # 反弹时，乘正数波动也行
            def_change *= random.uniform(1.0 - vol, 1.0 + vol)
            
        if str_change >= 0:
            str_change *= random.uniform(1.0 - vol, 1.0 + vol)
        else:
            str_change *= random.uniform(1.0 - vol, 1.0 + vol)

        # 应用变化 (防御减少，压力增加)
        self.defense = max(0.0, min(100.0, self.defense - def_change))
        self.stress = max(0.0, min(100.0, self.stress + str_change))

        # 返回结果 (注意：返回的change带符号，防御为负变化，压力为正变化)
        return InterrogationResult(
            round_num=self.current_round,
            defense=self.defense,
            stress=self.stress,
            strategy=strategy,
            defense_change=-def_change,  # 显示为负数
            stress_change=str_change
        )