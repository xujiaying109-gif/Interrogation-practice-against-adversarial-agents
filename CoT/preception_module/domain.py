from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any

class Intent(Enum):
    """审讯意图类型"""
    EVIDENCE = "evidence"
    TIME_INQUIRY = "time"
    LOCATION_INQUIRY = "location"
    RELATION_INQUIRY = "relation"
    PRESSURE = "pressure"
    CHIT_CHAT = "chit_chat"
    DETAIL_INQUIRY = "detail"
    TRAP = "trap"
    UNKNOWN = "unknown"

@dataclass
class PerceptionResult:
    """感知分析结果"""
    intent: Intent
    evidence_strength: float
    is_trap: bool
    normalized_entities: List[str]
    new_entities: List[str]
    keywords: List[str] = field(default_factory=list)
    confidence: float = 1.0
    query_type: str = "general"

    def to_dict(self):
        return {
            "intent": self.intent.value,
            "evidence_strength": self.evidence_strength,
            "is_trap": self.is_trap,
            "normalized_entities": self.normalized_entities,
            "new_entities": self.new_entities,
            "keywords": self.keywords,
            "confidence": self.confidence,
            "query_type": self.query_type
        }

class DeceptionType(Enum):
    DISTORT = "distort"
    OMIT = "omit"
    FABRICATE = "fabricate"
    RATIONALIZE = "rationalize"

class Participant:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "")
        self.role = kwargs.get("role", "")

class TruthEvent:
    def __init__(self, **kwargs):
        self.eid = kwargs.get("eid", "")
        self.type = kwargs.get("type", "")
        self.description = kwargs.get("description", "")
        self.participants = kwargs.get("participants", [])
        self.time = kwargs.get("time", "")
        self.location = kwargs.get("location", "")
        self.is_crime = kwargs.get("is_crime", False)
        self.crime_type = kwargs.get("crime_type", "")

        # 转换参与者
        if self.participants and isinstance(self.participants[0], dict):
            self.participants = [Participant(**p) for p in self.participants]

class FakeEvent:
    def __init__(self, **kwargs):
        self.eid = kwargs.get("eid", "")
        self.type = kwargs.get("type", "")
        self.description = kwargs.get("description", "")
        self.participants = kwargs.get("participants", [])
        self.anchor_to = kwargs.get("anchor_to", "")

        # 处理deception_type
        deception_type = kwargs.get("deception_type", "distort")
        if isinstance(deception_type, str):
            self.deception_type = DeceptionType(deception_type)
        else:
            self.deception_type = deception_type

        self.narrative = kwargs.get("narrative", "")
        self.confidence = kwargs.get("confidence", 0.5)

        # 转换参与者
        if self.participants and isinstance(self.participants[0], dict):
            self.participants = [Participant(**p) for p in self.participants]

class ContextEvent:
    def __init__(self, **kwargs):
        self.eid = kwargs.get("eid", "")
        self.type = kwargs.get("type", "")
        self.description = kwargs.get("description", "")
        self.participants = kwargs.get("participants", [])

        if self.participants and isinstance(self.participants[0], dict):
            self.participants = [Participant(**p) for p in self.participants]

@dataclass
class RetrievedKnowledge:
    """检索到的知识"""
    truth_events: List[TruthEvent]
    fake_events: List[FakeEvent]
    context_events: List[ContextEvent]

    def to_dict(self):
        return {
            "truth_count": len(self.truth_events),
            "fake_count": len(self.fake_events),
            "context_count": len(self.context_events)
        }

class EventGraph:
    """事件图谱"""

    def __init__(self, case_data=None):
        self.truth_events = []
        self.fake_events = []
        self.context_events = []

        if case_data:
            self.load_case(case_data)

    def load_case(self, case_data):
        """加载案件数据"""
        # 加载真实事件
        for event_data in case_data.get("truth_events", []):
            event = TruthEvent(**event_data)
            self.truth_events.append(event)

        # 加载虚假事件
        for event_data in case_data.get("fake_events", []):
            event = FakeEvent(**event_data)
            self.fake_events.append(event)

        # 加载上下文事件
        for event_data in case_data.get("context_events", []):
            event = ContextEvent(**event_data)
            self.context_events.append(event)

    def to_dict(self):
        return {
            "stats": {
                "total_events": len(self.truth_events) + len(self.fake_events) + len(self.context_events),
                "truth_events": len(self.truth_events),
                "fake_events": len(self.fake_events),
                "context_events": len(self.context_events)
            }
        }
