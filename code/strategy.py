"""
策略模块 - 决策引擎
"""
from enum import Enum
from dataclasses import dataclass
from log_utils import get_logger

logger = get_logger(__name__)


class Strategy(Enum):
    DIRECT_DENIAL = "direct_denial"
    FEIGN_IGNORANCE = "feign_ignorance"
    RATIONALIZATION = "rationalization"
    FULL_CONFESSION = "full_confession"


@dataclass
class PsychologicalState:
    defense_value: float
    stress_value: float
    profile_type: str = "arrogant"

    def to_dict(self):
        return {"defense": self.defense_value, "stress": self.stress_value}


@dataclass
class DeceptionStrategy:
    primary_strategy: Strategy
    reasoning: str = ""


class StrategyModule:
    def __init__(self, config):
        self.config = config

    def select_strategy(self, defense_value: float, stress_value: float,
                        intent: str, evidence_strength: float, is_trap: bool) -> DeceptionStrategy:
                        
        # 【核心重构】不再使用定死的 if-else，而是直接询问性格对象的偏好
        # config 对象在组装时必须包含 personality 实例
        strategy_str = self.config.personality.get_strategy_bias(
            defense_value, stress_value, intent, evidence_strength, is_trap
        )
        
        # 将字符串映射回强类型的 Enum
        strategy_enum = Strategy(strategy_str)
        
        reasoning = f"基于 {self.config.personality.name} 的底层倾向计算"
        if strategy_enum == Strategy.FULL_CONFESSION:
            reasoning = "心理防线彻底崩溃，全盘招供"
            
        return DeceptionStrategy(strategy_enum, reasoning)