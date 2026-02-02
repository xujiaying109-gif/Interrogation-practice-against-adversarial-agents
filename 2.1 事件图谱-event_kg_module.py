"""
事件图谱双层知识图谱模块
修复了 UUID 切片错误和 dataclass 问题
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import uuid
import re
from enum import Enum


# ==================== 数据结构定义 ====================

class EventType(Enum):
    """事件类型"""
    BRIBERY = "bribery"  # 受贿
    MEETING = "meeting"  # 见面
    TRANSFER = "transfer"  # 转账
    COMMUNICATION = "communication"  # 通讯
    DOCUMENT = "document"  # 文件
    LIFESTYLE = "lifestyle"  # 生活行为
    OTHER = "other"


class DeceptionType(Enum):
    """欺骗算子类型"""
    DISTORT = "distort"  # 扭曲
    OMIT = "omit"  # 删减
    FABRICATE = "fabricate"  # 捏造
    RATIONALIZE = "rationalize"  # 合理化


@dataclass
class Participant:
    """参与者"""
    name: str
    role: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)


# ==================== 修复：重新组织继承结构 ====================

@dataclass
class BaseEventCore:
    """基础事件核心字段（无默认值）"""
    eid: str
    type: EventType
    description: str
    participants: List[Participant]

    def get_participant_names(self) -> List[str]:
        return [p.name for p in self.participants]


@dataclass
class BaseEvent(BaseEventCore):
    """完整基础事件（有默认值）"""
    time: Optional[datetime] = None
    location: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TruthEvent(BaseEvent):
    """真实事件"""
    is_crime: bool = False
    crime_type: Optional[str] = None
    severity: float = 0.0


@dataclass
class FakeEvent(BaseEvent):
    """虚假事件"""
    # 注意：所有新加字段都有默认值
    anchor_to: str = ""
    deception_type: DeceptionType = DeceptionType.DISTORT
    narrative: str = ""
    confidence: float = 0.5
    risk_level: str = "medium"
    deception_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEvent(BaseEvent):
    """上下文事件（闲聊、临时记忆）"""
    is_crime_related: bool = False
    source_round: int = 0
    spontaneity: float = 0.5


# ==================== 欺骗算子引擎 ====================

class DeceptionOperatorEngine:
    """欺骗算子引擎"""

    def __init__(self):
        self.rules = []
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认规则"""
        # 受贿→借款
        self.rules.append({
            "pattern": {"type": "bribery", "has_amount": True},
            "operator": DeceptionType.DISTORT,
            "replacement": {
                "new_type": "transfer",
                "new_description": "资金借贷",
                "narrative": "是朋友间的资金周转"
            },
            "confidence": 0.9
        })
        # 删除中间人
        self.rules.append({
            "pattern": {"has_middleman": True},
            "operator": DeceptionType.OMIT,
            "replacement": {
                "remove_roles": ["middleman"],
                "narrative": "直接联系，没有中间人"
            },
            "confidence": 0.8
        })

    def apply_deception(self, truth_event: TruthEvent) -> FakeEvent:
        """应用欺骗算子"""
        # 匹配规则
        matched_rule = None
        for rule in self.rules:
            if self._matches_rule(truth_event, rule["pattern"]):
                matched_rule = rule
                break

        if not matched_rule:
            matched_rule = {
                "operator": DeceptionType.DISTORT,
                "replacement": {"narrative": "事情不是你想的那样"},
                "confidence": 0.5
            }

        # 创建虚假事件 - 修复 UUID 切片错误
        # 原错误: uuid.uuid4()[:4] - 不能对 UUID 对象切片
        # 修复: str(uuid.uuid4())[:4] 或 uuid.uuid4().hex[:4]
        fake_event = FakeEvent(
            eid=f"fake_{truth_event.eid}_{str(uuid.uuid4())[:8]}",  # 修复这里
            type=truth_event.type,
            description=matched_rule["replacement"].get("new_description", truth_event.description),
            participants=truth_event.participants.copy(),
            time=truth_event.time,
            location=truth_event.location,
            attributes=truth_event.attributes.copy(),
            anchor_to=truth_event.eid,  # 设置锚点
            deception_type=matched_rule["operator"],
            narrative=matched_rule["replacement"].get("narrative", ""),
            confidence=matched_rule["confidence"]
        )

        # 应用具体操作
        if matched_rule["operator"] == DeceptionType.OMIT:
            fake_event.participants = [
                p for p in fake_event.participants
                if p.role not in matched_rule["replacement"].get("remove_roles", [])
            ]

        return fake_event

    def _matches_rule(self, event: TruthEvent, pattern: Dict) -> bool:
        """检查是否匹配规则"""
        if "type" in pattern and event.type.value != pattern["type"]:
            return False

        if "has_amount" in pattern and pattern["has_amount"]:
            if "amount" not in event.attributes:
                return False

        if "has_middleman" in pattern and pattern["has_middleman"]:
            has_middleman = any(p.role == "middleman" for p in event.participants)
            if not has_middleman:
                return False

        return True


# ==================== 事件图谱核心类 ====================

class EventGraph:
    """双层事件图谱"""

    def __init__(self, case_data: Dict = None):
        # 三层存储
        self.truth_events: Dict[str, TruthEvent] = {}
        self.fake_events: Dict[str, FakeEvent] = {}
        self.context_events: Dict[str, ContextEvent] = {}

        # 索引
        self.entity_index: Dict[str, List[str]] = defaultdict(list)
        self.entities: Set[str] = set()

        # 引擎
        self.deception_engine = DeceptionOperatorEngine()

        # 加载数据
        if case_data:
            self.load_from_dict(case_data)

    def load_from_dict(self, case_data: Dict):
        """加载案件数据"""
        # 加载真实事件
        for event_dict in case_data.get("truth_events", []):
            event = self._dict_to_truth_event(event_dict)
            self.add_truth_event(event)

        # 加载预设虚假事件
        for event_dict in case_data.get("fake_events", []):
            event = self._dict_to_fake_event(event_dict)
            self.add_fake_event(event)

        # 自动生成缺失的虚假事件
        self._generate_missing_fake_events()

    def _dict_to_truth_event(self, data: Dict) -> TruthEvent:
        """字典转真实事件"""
        participants = [
            Participant(
                name=p["name"],
                role=p.get("role", "unknown"),
                attributes=p.get("attributes", {})
            )
            for p in data.get("participants", [])
        ]

        time = None
        if data.get("time"):
            try:
                time_str = data["time"].replace("Z", "+00:00")
                time = datetime.fromisoformat(time_str)
            except:
                pass

        return TruthEvent(
            eid=data["eid"],
            type=EventType(data["type"]),
            description=data["description"],
            participants=participants,
            time=time,
            location=data.get("location"),
            attributes=data.get("attributes", {}),
            is_crime=data.get("is_crime", False),
            crime_type=data.get("crime_type"),
            severity=data.get("severity", 0.0)
        )

    def _dict_to_fake_event(self, data: Dict) -> FakeEvent:
        """字典转虚假事件"""
        participants = [
            Participant(
                name=p["name"],
                role=p.get("role", "unknown"),
                attributes=p.get("attributes", {})
            )
            for p in data.get("participants", [])
        ]

        time = None
        if data.get("time"):
            try:
                time_str = data["time"].replace("Z", "+00:00")
                time = datetime.fromisoformat(time_str)
            except:
                pass

        return FakeEvent(
            eid=data["eid"],
            type=EventType(data["type"]),
            description=data["description"],
            participants=participants,
            time=time,
            location=data.get("location"),
            attributes=data.get("attributes", {}),
            anchor_to=data["anchor_to"],
            deception_type=DeceptionType(data["deception_type"]),
            narrative=data["narrative"],
            confidence=data.get("confidence", 0.5)
        )

    def add_truth_event(self, event: TruthEvent):
        """添加真实事件"""
        self.truth_events[event.eid] = event
        self._update_index(event)

    def add_fake_event(self, event: FakeEvent):
        """添加虚假事件"""
        self.fake_events[event.eid] = event
        self._update_index(event)

    def add_context_event(self, event: ContextEvent):
        """添加上下文事件"""
        self.context_events[event.eid] = event
        self._update_index(event)

    def _update_index(self, event: BaseEvent):
        """更新索引"""
        for participant in event.participants:
            self.entity_index[participant.name].append(event.eid)
            self.entities.add(participant.name)

    def _generate_missing_fake_events(self):
        """为没有虚假事件的真实事件生成虚假版本"""
        for truth_id, truth_event in self.truth_events.items():
            has_fake = any(fake.anchor_to == truth_id for fake in self.fake_events.values())
            if not has_fake and truth_event.is_crime:
                fake_event = self.deception_engine.apply_deception(truth_event)
                self.add_fake_event(fake_event)

    def get_entity_set(self) -> Set[str]:
        """获取所有实体"""
        return self.entities.copy()

    def retrieve_by_entities(self, entities: List[str], top_k: int = 5) -> Dict[str, List[BaseEvent]]:
        """根据实体检索事件"""
        result = {"truth": [], "fake": [], "context": []}
        seen_ids = set()

        for entity in entities:
            if entity in self.entity_index:
                for event_id in self.entity_index[entity][:top_k]:
                    if event_id in seen_ids:
                        continue
                    seen_ids.add(event_id)

                    if event_id in self.truth_events:
                        result["truth"].append(self.truth_events[event_id])
                    elif event_id in self.fake_events:
                        result["fake"].append(self.fake_events[event_id])
                    elif event_id in self.context_events:
                        result["context"].append(self.context_events[event_id])

        return result

    def create_context_event(self, description: str, participants: List[str], round_num: int) -> ContextEvent:
        """创建上下文事件"""
        # 修复这里的 UUID 切片错误
        participant_objs = [Participant(name=name, role="context") for name in participants]

        event = ContextEvent(
            eid=f"context_{str(uuid.uuid4())[:8]}",  # 修复这里
            type=EventType.LIFESTYLE,
            description=description,
            participants=participant_objs,
            time=datetime.now(),
            is_crime_related=False,
            source_round=round_num
        )

        self.add_context_event(event)
        return event

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "truth_events": [
                {
                    "eid": e.eid,
                    "type": e.type.value,
                    "description": e.description,
                    "participants": [{"name": p.name, "role": p.role} for p in e.participants],
                    "is_crime": e.is_crime
                }
                for e in self.truth_events.values()
            ],
            "fake_events": [
                {
                    "eid": e.eid,
                    "description": e.description,
                    "narrative": e.narrative,
                    "anchor_to": e.anchor_to,
                    "deception_type": e.deception_type.value
                }
                for e in self.fake_events.values()
            ],
            "context_events": [
                {"description": e.description, "source_round": e.source_round}
                for e in self.context_events.values()
            ],
            "stats": {
                "truth_count": len(self.truth_events),
                "fake_count": len(self.fake_events),
                "context_count": len(self.context_events),
                "entity_count": len(self.entities)
            }
        }


# ==================== 检索引擎 ====================

class EventRetrievalEngine:
    """事件检索引擎"""

    def __init__(self, event_graph: EventGraph):
        self.event_graph = event_graph

    def analyze_query(self, query: str) -> Dict:
        """分析查询语句"""
        analysis = {
            "intent": "unknown",
            "evidence_strength": 0.0,
            "entities": [],
            "is_trap": False
        }

        # 提取实体
        known_entities = self.event_graph.get_entity_set()
        for entity in known_entities:
            if entity in query:
                analysis["entities"].append(entity)

        # 分析意图
        query_lower = query.lower()
        if any(word in query_lower for word in ["证据", "监控", "流水", "记录"]):
            analysis["intent"] = "evidence"
            analysis["evidence_strength"] = 0.8
        elif any(word in query_lower for word in ["时间", "时候", "日期"]):
            analysis["intent"] = "time"
        elif any(word in query_lower for word in ["地点", "位置", "在哪"]):
            analysis["intent"] = "location"
        elif any(word in query_lower for word in ["解释", "交代", "承认"]):
            analysis["intent"] = "pressure"
            analysis["evidence_strength"] = 0.6

        # 检测陷阱
        trap_keywords = ["真的吗", "你确定", "我听说", "有人看到", "别撒谎"]
        if any(keyword in query_lower for keyword in trap_keywords):
            analysis["is_trap"] = True
            analysis["evidence_strength"] = max(analysis["evidence_strength"], 0.5)

        return analysis

    def retrieve_with_psychology(self, query: str, psych_state: Dict) -> Dict:
        """考虑心理状态的检索"""
        analysis = self.analyze_query(query)
        defense = psych_state.get("defense_value", 50)
        stress = psych_state.get("stress_value", 50)

        # 基础检索
        retrieved = self.event_graph.retrieve_by_entities(analysis["entities"])

        # 根据心理状态过滤
        filtered = {"truth": [], "fake": [], "context": []}

        for layer, events in retrieved.items():
            for event in events:
                # 高压状态下减少真实事件暴露
                if layer == "truth" and stress > 70:
                    if isinstance(event, TruthEvent) and not event.is_crime:
                        filtered[layer].append(event)
                # 低防御状态下使用高置信度的虚假事件
                elif layer == "fake" and defense < 40:
                    if isinstance(event, FakeEvent) and event.confidence > 0.7:
                        filtered[layer].append(event)
                else:
                    filtered[layer].append(event)

        return {
            "events": filtered,
            "analysis": analysis,
            "query": query
        }

    def generate_deception_guidance(self, retrieved_events: Dict, psych_state: Dict) -> Dict:
        """生成欺骗指导"""
        guidance = {
            "strategy": "default",
            "focus_areas": [],
            "avoid_topics": [],
            "verbal_cues": []
        }

        # 分析真实事件中的敏感点
        truth_events = retrieved_events.get("truth", [])
        crime_events = [e for e in truth_events if getattr(e, 'is_crime', False)]

        if crime_events:
            guidance["strategy"] = "high_risk"
            guidance["focus_areas"].append("crime_denial")
            for event in crime_events[:2]:
                guidance["avoid_topics"].append(event.description)

        # 根据心理状态调整
        stress = psych_state.get("stress_value", 50)
        if stress > 70:
            guidance["strategy"] = "conservative"
            guidance["verbal_cues"].extend(["使用模糊语言", "避免直接否认"])

        return guidance


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("事件图谱模块测试...")

    # 测试数据
    test_case = {
        "truth_events": [
            {
                "eid": "test_001",
                "type": "bribery",
                "description": "测试受贿事件",
                "participants": [
                    {"name": "张局长", "role": "suspect"},
                    {"name": "李某", "role": "briber"}
                ],
                "is_crime": True
            }
        ]
    }

    # 创建事件图谱
    try:
        eg = EventGraph(test_case)
        print(f"✅ 创建成功！")
        print(f"   真实事件数: {len(eg.truth_events)}")
        print(f"   虚假事件数: {len(eg.fake_events)}")

        # 测试检索
        retrieved = eg.retrieve_by_entities(["张局长"])
        print(f"✅ 检索测试:")
        print(f"   检索到 {len(retrieved['fake'])} 个虚假事件")

        for fake in retrieved["fake"]:
            print(f"   虚假事件: {fake.description}")
            print(f"     欺骗叙述: {fake.narrative}")
            print(f"     置信度: {fake.confidence}")

        # 测试上下文事件
        ctx = eg.create_context_event("嫌疑人提到喜欢喝茶", ["张局长"], 1)
        print(f"✅ 上下文事件测试:")
        print(f"   创建的上下文事件: {ctx.description}")

        # 测试检索引擎
        engine = EventRetrievalEngine(eg)
        query = "张局长和李某是什么关系？"
        analysis = engine.analyze_query(query)
        print(f"✅ 检索引擎测试:")
        print(f"   查询: {query}")
        print(f"   意图: {analysis['intent']}")
        print(f"   实体: {analysis['entities']}")

        print("\n🎉 所有测试通过！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
