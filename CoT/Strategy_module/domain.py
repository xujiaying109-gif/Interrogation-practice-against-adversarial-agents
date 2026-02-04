from enum import Enum
from dataclasses import dataclass, field
from typing import List

class Strategy(Enum):
    """对抗策略类型"""
    DIRECT_DENIAL = "direct_denial"
    FEIGN_IGNORANCE = "feign_ignorance"
    RATIONALIZATION = "rationalization"
    RED_HERRING = "red_herring"
    PARTIAL_ADMISSION = "partial_admission"
    FULL_CONFESSION = "full_confession"
    INFORMATION_DILUTION = "information_dilution"
    DEFLECT = "deflect"
    COUNTER_ATTACK = "counter_attack"

class StatusLabel(Enum):
    """心理状态标签"""
    CALM = "CALM"
    DEFENSIVE = "DEFENSIVE"
    ANXIOUS = "ANXIOUS"
    HESITANT = "HESITANT"
    BROKEN = "BROKEN"

@dataclass
class PsychologicalState:
    """心理状态"""
    defense_value: float
    stress_value: float
    status_label: StatusLabel
    profile_type: str = "Arrogant"

    def to_dict(self):
        return {
            "defense_value": self.defense_value,
            "stress_value": self.stress_value,
            "status_label": self.status_label.value,
            "profile_type": self.profile_type
        }

@dataclass
class DeceptionStrategy:
    """欺骗策略"""
    primary_strategy: Strategy
    secondary_strategy: Strategy = None
    focus_areas: List[str] = field(default_factory=list)
    avoid_topics: List[str] = field(default_factory=list)
    verbal_cues: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    confidence: float = 0.8

    def to_dict(self):
        result = {
            "primary_strategy": self.primary_strategy.value,
            "secondary_strategy": self.secondary_strategy.value if self.secondary_strategy else None,
            "focus_areas": self.focus_areas,
            "avoid_topics": self.avoid_topics,
            "verbal_cues": self.verbal_cues,
            "risk_level": self.risk_level,
            "confidence": self.confidence
        }
        return result
