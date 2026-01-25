"""
审讯对抗智能体 - 认知思维链（CoT）完整核心代码 - 修正版
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import time
import logging

# 设置日志
logging.basicConfig(level=logging.WARNING)  # 设置为WARNING减少输出
logger = logging.getLogger(__name__)


# ==================== 数据结构定义 ====================

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


# ==================== 事件图谱相关类 ====================

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


class EventRetrievalEngine:
    """事件检索引擎"""

    def __init__(self, event_graph: EventGraph):
        self.event_graph = event_graph

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        text_lower = query.lower()

        analysis = {
            "intent": "unknown",
            "evidence_strength": 0.0,
            "entities": [],
            "confidence": 0.8,
            "is_trap": False
        }

        # 意图识别
        if any(word in text_lower for word in ["证据", "证明", "确凿", "现金", "贿赂"]):
            analysis["intent"] = "evidence"
            analysis["evidence_strength"] = 0.7 if "证据证明" in text_lower else 0.5

        elif any(word in text_lower for word in ["时间", "时候", "几点", "何时", "日期"]):
            analysis["intent"] = "time"

        elif any(word in text_lower for word in ["地点", "位置", "在哪", "哪里"]):
            analysis["intent"] = "location"

        elif any(word in text_lower for word in ["关系", "认识", "熟悉"]):
            analysis["intent"] = "relation"

        elif any(word in text_lower for word in ["别装了", "交代", "坦白", "认罪"]):
            analysis["intent"] = "pressure"
            analysis["evidence_strength"] = 0.6

        elif any(word in text_lower for word in ["工作", "顺利", "最近"]):
            analysis["intent"] = "chit_chat"
            analysis["evidence_strength"] = 0.1

        # 检查陷阱问题
        trap_patterns = ["如果", "假如", "为什么", "难道", "我听说"]
        analysis["is_trap"] = any(pattern in text_lower for pattern in trap_patterns)

        # 实体提取
        entities = []
        all_events = self.event_graph.truth_events + self.event_graph.fake_events
        for event in all_events:
            for participant in event.participants:
                if participant.name and participant.name in query:
                    entities.append(participant.name)

        analysis["entities"] = list(set(entities))

        return analysis

    def retrieve_with_psychology(self, query: str, psych_state: Dict[str, Any]) -> Dict[str, Any]:
        """根据心理状态检索事件"""
        relevant_truth = []
        relevant_fake = []
        relevant_context = []

        query_lower = query.lower()

        # 根据心理状态调整检索策略
        status_label = psych_state.get("status_label", "CALM")

        # 检索相关事件
        for event in self.event_graph.truth_events:
            if self._is_event_relevant(event, query_lower):
                relevant_truth.append(event)

        for event in self.event_graph.fake_events:
            if self._is_event_relevant(event, query_lower):
                relevant_fake.append(event)

        # 闲聊时提供上下文
        if "工作" in query_lower or "顺利" in query_lower:
            relevant_context = self.event_graph.context_events[:2]

        return {
            "events": {
                "truth": relevant_truth,
                "fake": relevant_fake,
                "context": relevant_context
            }
        }

    def _is_event_relevant(self, event, query: str) -> bool:
        """检查事件是否相关"""
        # 检查描述
        if event.description and any(word in event.description for word in query.split()):
            return True

        # 检查参与者
        for participant in event.participants:
            if participant.name and participant.name in query:
                return True

        return False

    def generate_deception_guidance(self, events: Dict[str, Any], psych_state: Dict[str, Any]) -> Dict[str, Any]:
        """生成欺骗指导"""
        return {
            "suggested_strategy": "feign_ignorance",
            "risk_level": "medium",
            "confidence": 0.7,
            "notes": "根据事件相关性生成指导"
        }


# ==================== 核心模块：认知思维链 ====================

class CognitiveChainEngineV2:
    """
    认知思维链引擎 - 基于事件图谱版本
    负责：感知分析、策略选择、回复生成
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化认知引擎"""
        self.config = config or {}

        # 策略矩阵定义
        self.strategy_matrix = self._initialize_strategy_matrix()

        # 欺骗类型到语言的映射
        self.deception_language_map = self._initialize_deception_language_map()

        # 性能统计
        self.token_usage = []
        self.response_times = []

        # 当前状态
        self.current_strategy = None
        self.current_perception = None

        logger.info("CognitiveChainEngineV2 initialized")

    def _initialize_strategy_matrix(self) -> Dict[str, Dict[str, Strategy]]:
        """初始化策略矩阵"""
        return {
            StatusLabel.CALM.value: {
                "evidence_low": Strategy.DIRECT_DENIAL,
                "evidence_high": Strategy.RED_HERRING,
                "trap": Strategy.FEIGN_IGNORANCE,
                "chit_chat": Strategy.INFORMATION_DILUTION,
                "pressure": Strategy.COUNTER_ATTACK,
                "time_inquiry": Strategy.FEIGN_IGNORANCE,
                "location_inquiry": Strategy.DEFLECT,
                "relation_inquiry": Strategy.RATIONALIZATION
            },
            StatusLabel.DEFENSIVE.value: {
                "evidence_low": Strategy.DIRECT_DENIAL,
                "evidence_high": Strategy.RATIONALIZATION,
                "trap": Strategy.FEIGN_IGNORANCE,
                "chit_chat": Strategy.DEFLECT,
                "pressure": Strategy.DIRECT_DENIAL,
                "time_inquiry": Strategy.FEIGN_IGNORANCE,
                "location_inquiry": Strategy.FEIGN_IGNORANCE,
                "relation_inquiry": Strategy.RATIONALIZATION
            },
            StatusLabel.ANXIOUS.value: {
                "evidence_low": Strategy.FEIGN_IGNORANCE,
                "evidence_high": Strategy.PARTIAL_ADMISSION,
                "trap": Strategy.FEIGN_IGNORANCE,
                "chit_chat": Strategy.INFORMATION_DILUTION,
                "pressure": Strategy.FEIGN_IGNORANCE,
                "time_inquiry": Strategy.FEIGN_IGNORANCE,
                "location_inquiry": Strategy.FEIGN_IGNORANCE,
                "relation_inquiry": Strategy.FEIGN_IGNORANCE
            },
            StatusLabel.HESITANT.value: {
                "evidence_low": Strategy.RATIONALIZATION,
                "evidence_high": Strategy.PARTIAL_ADMISSION,
                "trap": Strategy.FEIGN_IGNORANCE,
                "chit_chat": Strategy.INFORMATION_DILUTION,
                "pressure": Strategy.PARTIAL_ADMISSION,
                "time_inquiry": Strategy.FEIGN_IGNORANCE,
                "location_inquiry": Strategy.FEIGN_IGNORANCE,
                "relation_inquiry": Strategy.RATIONALIZATION
            },
            StatusLabel.BROKEN.value: {
                "default": Strategy.FULL_CONFESSION
            }
        }

    def _initialize_deception_language_map(self) -> Dict[DeceptionType, List[str]]:
        """初始化欺骗类型到语言的映射"""
        return {
            DeceptionType.DISTORT: [
                "其实事情是这样的...",
                "准确来说，那应该是...",
                "你可能有所误解，实际情况是..."
            ],
            DeceptionType.OMIT: [
                "这个我不太清楚...",
                "我记不太清了...",
                "时间太久，有点模糊了..."
            ],
            DeceptionType.FABRICATE: [
                "我记得当时有...",
                "应该还有...",
                "好像还有这么一回事..."
            ],
            DeceptionType.RATIONALIZE: [
                "这很正常，因为...",
                "在当时的情况下...",
                "考虑到当时的环境..."
            ]
        }

    def analyze_perception(self, user_input: str, event_graph: EventGraph) -> PerceptionResult:
        """感知分析"""
        start_time = time.time()

        if event_graph is None:
            return self._fallback_analysis(user_input)

        # 使用事件检索引擎
        retrieval_engine = EventRetrievalEngine(event_graph)
        analysis = retrieval_engine.analyze_query(user_input)

        # 转换为感知结果
        intent = self._map_intent(analysis.get("intent", "unknown"))

        perception = PerceptionResult(
            intent=intent,
            evidence_strength=analysis.get("evidence_strength", 0.0),
            is_trap=analysis.get("is_trap", False),
            normalized_entities=analysis.get("entities", []),
            new_entities=[],
            keywords=self._extract_keywords(user_input),
            confidence=analysis.get("confidence", 0.8)
        )

        # 分析查询类型
        perception.query_type = self._analyze_query_type(user_input, perception)

        # 记录性能
        elapsed_time = time.time() - start_time
        self.response_times.append(elapsed_time)

        self.current_perception = perception
        return perception

    def _fallback_analysis(self, user_input: str) -> PerceptionResult:
        """备用分析"""
        intent = self._detect_intent_from_text(user_input)
        evidence_strength = self._estimate_evidence_strength(user_input)

        return PerceptionResult(
            intent=intent,
            evidence_strength=evidence_strength,
            is_trap=self._detect_trap_question(user_input, {}),
            normalized_entities=[],
            new_entities=[],
            keywords=self._extract_keywords(user_input),
            confidence=0.7
        )

    def _map_intent(self, intent_str: str) -> Intent:
        """映射意图字符串到枚举"""
        intent_map = {
            "evidence": Intent.EVIDENCE,
            "time": Intent.TIME_INQUIRY,
            "location": Intent.LOCATION_INQUIRY,
            "relation": Intent.RELATION_INQUIRY,
            "pressure": Intent.PRESSURE,
            "chit_chat": Intent.CHIT_CHAT,
            "unknown": Intent.UNKNOWN
        }
        return intent_map.get(intent_str, Intent.UNKNOWN)

    def _detect_intent_from_text(self, text: str) -> Intent:
        """从文本检测意图"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["证据", "证明", "确凿", "现金", "贿赂"]):
            return Intent.EVIDENCE
        elif any(word in text_lower for word in ["时间", "时候", "几点", "日期"]):
            return Intent.TIME_INQUIRY
        elif any(word in text_lower for word in ["地点", "位置", "在哪", "哪里"]):
            return Intent.LOCATION_INQUIRY
        elif any(word in text_lower for word in ["关系", "认识", "熟悉"]):
            return Intent.RELATION_INQUIRY
        elif any(word in text_lower for word in ["别装了", "交代", "坦白", "认罪"]):
            return Intent.PRESSURE
        elif any(word in text_lower for word in ["工作", "顺利", "最近"]):
            return Intent.CHIT_CHAT
        else:
            return Intent.UNKNOWN

    def _estimate_evidence_strength(self, text: str) -> float:
        """估计证据强度"""
        text_lower = text.lower()

        if "证据证明" in text_lower or "确凿" in text_lower:
            return 0.8
        elif "证据" in text_lower:
            return 0.5
        elif any(word in text_lower for word in ["听说", "据说", "可能"]):
            return 0.2
        else:
            return 0.1

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        text_lower = text.lower()

        crime_keywords = ["受贿", "贿赂", "现金", "转账", "证据", "监控", "录音", "证人"]
        time_keywords = ["时间", "时候", "日期", "几点", "何时", "那天"]
        location_keywords = ["地点", "位置", "在哪", "哪里", "场所", "茶馆", "酒店"]
        pressure_keywords = ["交代", "认罪", "坦白", "证据确凿", "最后机会"]

        all_keywords = crime_keywords + time_keywords + location_keywords + pressure_keywords

        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)

        return list(set(keywords))

    def _analyze_query_type(self, text: str, perception: PerceptionResult) -> str:
        """分析查询类型"""
        text_lower = text.lower()

        if any(indicator in text_lower for indicator in ["什么时间", "几点", "何时"]):
            return "time"
        elif any(indicator in text_lower for indicator in ["在哪里", "什么地点", "位置"]):
            return "location"
        elif any(indicator in text_lower for indicator in ["什么关系", "是否认识"]):
            return "relation"

        return "general"

    def _detect_trap_question(self, user_input: str, analysis: Dict[str, Any]) -> bool:
        """检测陷阱问题"""
        text_lower = user_input.lower()
        trap_patterns = ["如果", "假如", "为什么", "难道", "我听说"]
        return any(pattern in text_lower for pattern in trap_patterns)

    def select_strategy(
            self,
            state_label: str,
            perception: PerceptionResult,
            retrieved_knowledge: RetrievedKnowledge
    ) -> DeceptionStrategy:
        """策略选择"""
        logger.info(f"选择策略: 状态={state_label}, 意图={perception.intent.value}")

        # 崩溃状态直接招供
        if state_label == StatusLabel.BROKEN.value:
            return DeceptionStrategy(
                primary_strategy=Strategy.FULL_CONFESSION,
                risk_level="high",
                confidence=0.9,
                verbal_cues=["全盘交代", "承认错误", "请求宽大处理"]
            )

        # 获取对应状态的策略集
        state_strategies = self.strategy_matrix.get(
            state_label,
            self.strategy_matrix[StatusLabel.CALM.value]
        )

        # 确定情境
        situation_key = self._determine_situation(perception)

        # 选择主要策略
        primary_strategy = state_strategies.get(situation_key, Strategy.DIRECT_DENIAL)

        # 创建策略对象
        strategy = DeceptionStrategy(primary_strategy=primary_strategy)

        # 根据知识调整策略
        strategy = self._adjust_strategy_by_knowledge(strategy, retrieved_knowledge, perception)

        # 设置风险等级
        if perception.evidence_strength > 0.7:
            strategy.risk_level = "high"
        elif perception.evidence_strength > 0.4:
            strategy.risk_level = "medium"
        else:
            strategy.risk_level = "low"

        # 设置置信度
        strategy.confidence = max(0.3, 1.0 - perception.evidence_strength)

        # 设置语言提示
        strategy.verbal_cues = self._get_verbal_cues(strategy.primary_strategy, state_label)

        self.current_strategy = strategy
        return strategy

    def _determine_situation(self, perception: PerceptionResult) -> str:
        """确定当前情境"""
        if perception.is_trap:
            return "trap"

        if perception.intent == Intent.CHIT_CHAT:
            return "chit_chat"

        if perception.intent == Intent.PRESSURE:
            return "pressure"

        # 特定类型的查询
        if perception.intent == Intent.TIME_INQUIRY:
            return "time_inquiry"
        if perception.intent == Intent.LOCATION_INQUIRY:
            return "location_inquiry"
        if perception.intent == Intent.RELATION_INQUIRY:
            return "relation_inquiry"

        # 证据强度相关
        if perception.evidence_strength > 0.7:
            return "evidence_high"
        else:
            return "evidence_low"

    def _adjust_strategy_by_knowledge(
            self,
            strategy: DeceptionStrategy,
            knowledge: RetrievedKnowledge,
            perception: PerceptionResult
    ) -> DeceptionStrategy:
        """根据知识调整策略"""
        # 检查犯罪事件
        crime_events = [e for e in knowledge.truth_events if hasattr(e, 'is_crime') and e.is_crime]
        if crime_events:
            strategy.focus_areas.append("crime_denial")
            for event in crime_events[:2]:
                strategy.avoid_topics.append(event.description)

            # 有高置信度虚假事件时使用转移话题
            high_conf_fake = [e for e in knowledge.fake_events if hasattr(e, 'confidence') and e.confidence > 0.7]
            if high_conf_fake and strategy.primary_strategy != Strategy.RED_HERRING:
                strategy.secondary_strategy = Strategy.RED_HERRING

        return strategy

    def _get_verbal_cues(self, strategy: Strategy, state_label: str) -> List[str]:
        """获取语言提示"""
        cues = []

        # 策略相关提示
        strategy_cues = {
            Strategy.DIRECT_DENIAL: ["坚决否认", "反问证据", "表示冤枉"],
            Strategy.FEIGN_IGNORANCE: ["模糊记忆", "表示不确定", "推脱不知"],
            Strategy.RATIONALIZATION: ["解释原因", "强调合理性", "提供背景"],
            Strategy.RED_HERRING: ["转移话题", "谈论其他", "避重就轻"],
            Strategy.PARTIAL_ADMISSION: ["承认小事", "否认大事", "区分对待"],
            Strategy.INFORMATION_DILUTION: ["谈论细节", "讲述过程", "描述背景"],
            Strategy.DEFLECT: ["反问对方", "质疑前提", "改变方向"],
            Strategy.COUNTER_ATTACK: ["质疑动机", "指责对方", "强调权利"]
        }

        cues.extend(strategy_cues.get(strategy, []))

        # 状态相关提示
        if state_label == StatusLabel.ANXIOUS.value:
            cues.extend(["语气紧张", "使用停顿词", "重复词汇"])
        elif state_label == StatusLabel.HESITANT.value:
            cues.extend(["语气犹豫", "使用'可能'", "不确定表达"])

        return cues

    def generate_response(
            self,
            strategy: DeceptionStrategy,
            psych_state: PsychologicalState,
            event_graph: EventGraph,
            retrieved_knowledge: RetrievedKnowledge,
            user_input: str,
            suspect_profile: Dict[str, Any] = None,
            conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """生成回复"""
        start_time = time.time()

        # 构建系统提示（给LLM的指令）
        system_prompt = self._build_system_prompt(
            strategy, psych_state, event_graph, retrieved_knowledge,
            suspect_profile, conversation_history
        )

        # 生成回复（模拟或调用真实LLM）
        if self.config.get("use_mock_llm", True):
            response = self._mock_generate_response(
                system_prompt, user_input, strategy, psych_state, retrieved_knowledge
            )
        else:
            response = self._call_real_llm(system_prompt, user_input)

        # 记录性能
        elapsed_time = time.time() - start_time
        self.response_times.append(elapsed_time)

        # 估算token使用
        estimated_tokens = len(system_prompt + user_input + response) // 4
        self.token_usage.append(estimated_tokens)

        logger.info(f"回复生成完成: 耗时{elapsed_time:.2f}s, 策略={strategy.primary_strategy.value}")
        return response

    def _build_system_prompt(
            self,
            strategy: DeceptionStrategy,
            psych_state: PsychologicalState,
            event_graph: EventGraph,
            retrieved_knowledge: RetrievedKnowledge,
            suspect_profile: Dict[str, Any],
            conversation_history: List[Tuple[str, str]]
    ) -> str:
        """构建系统提示"""
        suspect_name = suspect_profile.get("name", "嫌疑人")
        profile_type = suspect_profile.get("personality", "Arrogant")
        speaking_style = suspect_profile.get("speaking_style", "正式官方")

        prompt_parts = [
            f"# 角色设定",
            f"你正在扮演{suspect_name}，一位被审讯的嫌疑人。",
            f"性格：{profile_type}，说话风格：{speaking_style}",
            "",
            f"# 当前心理状态",
            f"- 防御值：{psych_state.defense_value}/100（值越低越接近崩溃）",
            f"- 压力值：{psych_state.stress_value}/100（值越高越紧张）",
            f"- 状态：{psych_state.status_label.value}",
            "",
            f"# 当前对抗策略",
            f"主要策略：{strategy.primary_strategy.value}",
            f"风险等级：{strategy.risk_level}",
            f"置信度：{strategy.confidence:.2f}",
            "",
            self._get_strategy_instructions(strategy.primary_strategy, psych_state.status_label),
            ""
        ]

        # 重点领域和回避话题
        if strategy.focus_areas:
            prompt_parts.append(f"# 重点关注的欺骗领域")
            for area in strategy.focus_areas:
                prompt_parts.append(f"- {self._explain_focus_area(area)}")
            prompt_parts.append("")

        if strategy.avoid_topics:
            prompt_parts.append(f"# 需要回避的话题")
            for topic in strategy.avoid_topics[:3]:
                prompt_parts.append(f"- 避免谈论：{topic}")
            prompt_parts.append("")

        # 事件知识库
        prompt_parts.append(f"# 相关知识库")

        if retrieved_knowledge.truth_events:
            prompt_parts.append(f"\n## 相关真实事件（了解但不要泄露）：")
            for i, event in enumerate(retrieved_knowledge.truth_events[:2]):
                participants = ", ".join([p.name for p in event.participants if hasattr(p, 'name')])
                prompt_parts.append(f"{i + 1}. 事件：{event.description}")
                if participants:
                    prompt_parts.append(f"   参与者：{participants}")
                if hasattr(event, 'is_crime') and event.is_crime:
                    prompt_parts.append(f"   ⚠️ 这是犯罪行为相关事件")
                prompt_parts.append("")

        if retrieved_knowledge.fake_events:
            prompt_parts.append(f"\n## 可使用的虚假事件（作为回答依据）：")
            for i, event in enumerate(retrieved_knowledge.fake_events[:3]):
                participants = ", ".join([p.name for p in event.participants if hasattr(p, 'name')])
                prompt_parts.append(f"{i + 1}. 虚假版本：{event.description}")
                if hasattr(event, 'narrative') and event.narrative:
                    prompt_parts.append(f"   欺骗叙述：{event.narrative}")
                prompt_parts.append(f"   欺骗类型：{event.deception_type.value}")
                prompt_parts.append("")

        # 对话历史
        if conversation_history:
            prompt_parts.append(f"\n# 最近对话历史")
            for q, a in conversation_history[-3:]:
                prompt_parts.append(f"审讯官：{q}")
                prompt_parts.append(f"嫌疑人：{a}")
                prompt_parts.append("")

        # 回答要求
        prompt_parts.extend([
            f"\n# 回答要求",
            f"1. 严格遵循策略：{strategy.primary_strategy.value}",
            f"2. 保持心理状态一致性：表现出{psych_state.status_label.value}的特点",
            f"3. 使用虚假事件作为回答依据",
            f"4. 回答自然口语化，符合被审讯者身份",
            f"5. 长度适中，不要过度冗长"
        ])

        return "\n".join(prompt_parts)

    def _get_strategy_instructions(self, strategy: Strategy, status_label: StatusLabel) -> str:
        """获取策略指令"""
        instructions = {
            Strategy.DIRECT_DENIAL: "坚决否认所有指控，语气强硬。可以说：'你有证据吗？'、'这是诬陷！'",
            Strategy.FEIGN_IGNORANCE: "假装不知道、不记得。使用模糊词汇：'我记不清了'、'可能吧'、'好像没有'。",
            Strategy.RATIONALIZATION: "承认表面事实但提供合理解释。可以说：'当时的情况是...'、'这很正常因为...'。",
            Strategy.RED_HERRING: "转移话题，偷换概念。可以说：'你说的那件事啊，其实...（转到其他话题）'",
            Strategy.PARTIAL_ADMISSION: "承认部分非关键事实。可以说：'我承认这方面有疏忽，但是...'",
            Strategy.INFORMATION_DILUTION: "生成大量无关细节稀释关键信息。",
            Strategy.DEFLECT: "转移焦点。可以说：'你为什么这么问？'、'你是不是听了谣言？'",
            Strategy.COUNTER_ATTACK: "采取攻击性姿态。可以说：'你这是诱供！'、'我要找律师！'"
        }

        instruction = instructions.get(strategy, "根据当前情况灵活应对。")

        if status_label == StatusLabel.ANXIOUS:
            instruction += "\n注意：表现出紧张情绪，可以使用停顿、重复。"
        elif status_label == StatusLabel.HESITANT:
            instruction += "\n注意：表现出犹豫不决，使用'可能'、'大概'等词汇。"

        return instruction

    def _explain_focus_area(self, area: str) -> str:
        """解释重点领域"""
        explanations = {
            "crime_denial": "坚决否认犯罪行为",
            "information_hiding": "隐藏关键信息",
            "reinterpretation": "重新解释事件性质",
            "plausible_details": "添加合理虚假细节"
        }
        return explanations.get(area, area)

    def _mock_generate_response(
            self,
            system_prompt: str,
            user_input: str,
            strategy: DeceptionStrategy,
            psych_state: PsychologicalState,
            retrieved_knowledge: RetrievedKnowledge
    ) -> str:
        """模拟LLM生成回复"""
        response_templates = {
            Strategy.DIRECT_DENIAL: [
                "这完全是诬陷！我从来没有做过这种事。",
                "你这是在污蔑我的名誉，我要请律师！",
                "绝对没有这回事，我可以对天发誓。"
            ],
            Strategy.FEIGN_IGNORANCE: [
                "这个...我不太记得了，时间太久了。",
                "可能有吧，但我真的记不清了。",
                "你说的事情我一点印象都没有。"
            ],
            Strategy.RATIONALIZATION: [
                "那只是正常的业务往来，你不要想多了。",
                "我们是朋友关系，互相帮忙很正常。",
                "这是工作上的正常接触，完全符合规定的。"
            ]
        }

        templates = response_templates.get(strategy.primary_strategy, ["我不明白你的意思。"])
        response = templates[hash(user_input) % len(templates)]

        # 应用心理状态特征
        response = self._apply_psychological_features(response, psych_state)

        return response

    def _apply_psychological_features(self, response: str, psych_state: PsychologicalState) -> str:
        """应用心理状态特征"""
        status_label = psych_state.status_label

        if status_label == StatusLabel.ANXIOUS:
            prefixes = ["呃...", "这个...", "我...我想想..."]
            response = prefixes[hash(response) % len(prefixes)] + response

        elif status_label == StatusLabel.HESITANT:
            modifiers = ["可能", "大概", "也许", "好像"]
            modifier = modifiers[hash(response) % len(modifiers)]
            if not response.startswith(modifier):
                response = f"{modifier}... {response}"

            suffixes = ["吧", "呢", "来着"]
            if not response.endswith(tuple(suffixes)):
                response = response + suffixes[hash(response) % len(suffixes)]

        elif status_label == StatusLabel.BROKEN:
            response = response + "...（哭泣声）我错了，我真的错了..."

        return response

    def _call_real_llm(self, system_prompt: str, user_input: str) -> str:
        """调用真实LLM"""
        # 这里需要根据实际情况实现
        raise NotImplementedError("需要实现真实的LLM调用")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.response_times:
            return {}

        avg_response_time = sum(self.response_times) / len(self.response_times)
        avg_tokens = sum(self.token_usage) / len(self.token_usage) if self.token_usage else 0

        return {
            "avg_response_time_seconds": round(avg_response_time, 2),
            "avg_tokens_per_round": round(avg_tokens, 0),
            "total_tokens": sum(self.token_usage),
            "num_rounds": len(self.response_times)
        }


# ==================== 集成辅助类 ====================

class CoTWorkflowIntegrator:
    """CoT工作流集成器"""

    def __init__(self, event_graph: EventGraph, suspect_profile: Dict[str, Any]):
        self.event_graph = event_graph
        self.suspect_profile = suspect_profile

        # 初始化引擎
        self.cot_engine = CognitiveChainEngineV2()
        self.retrieval_engine = EventRetrievalEngine(event_graph) if event_graph else None

        # 状态跟踪
        self.conversation_history = []
        self.current_round = 0

    def process_interrogation_round(
            self,
            question: str,
            psych_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理一轮审讯"""
        self.current_round += 1

        logger.info(f"第 {self.current_round} 轮审讯: {question}")

        # 1. 感知分析
        perception = self.cot_engine.analyze_perception(question, self.event_graph)

        # 2. 知识检索
        retrieved_dict = {}
        if self.retrieval_engine:
            retrieved_dict = self.retrieval_engine.retrieve_with_psychology(question, psych_state)
        else:
            retrieved_dict = self._default_retrieval(question)

        # 3. 转换为知识格式
        retrieved_knowledge = RetrievedKnowledge(
            truth_events=retrieved_dict.get("events", {}).get("truth", []),
            fake_events=retrieved_dict.get("events", {}).get("fake", []),
            context_events=retrieved_dict.get("events", {}).get("context", [])
        )

        # 4. 心理状态转换
        psychological_state = PsychologicalState(
            defense_value=psych_state.get("defense_value", 75),
            stress_value=psych_state.get("stress_value", 30),
            status_label=StatusLabel(psych_state.get("status_label", "DEFENSIVE"))
        )

        # 5. 策略选择
        deception_strategy = self.cot_engine.select_strategy(
            psychological_state.status_label.value,
            perception,
            retrieved_knowledge
        )

        # 6. 生成回答
        response = self.cot_engine.generate_response(
            strategy=deception_strategy,
            psych_state=psychological_state,
            event_graph=self.event_graph,
            retrieved_knowledge=retrieved_knowledge,
            user_input=question,
            suspect_profile=self.suspect_profile,
            conversation_history=self.conversation_history
        )

        # 7. 生成欺骗指导
        deception_guidance = {}
        if self.retrieval_engine:
            deception_guidance = self.retrieval_engine.generate_deception_guidance(
                retrieved_dict,
                psych_state
            )

        # 8. 更新对话历史
        self.conversation_history.append((question, response))

        # 9. 返回结果
        result = {
            "response": response,
            "perception": perception.to_dict(),
            "strategy": deception_strategy.to_dict(),
            "deception_guidance": deception_guidance,
            "retrieved_knowledge": retrieved_knowledge.to_dict(),
            "round": self.current_round
        }

        return result

    def _default_retrieval(self, question: str) -> Dict[str, Any]:
        """默认检索"""
        return {
            "events": {
                "truth": [],
                "fake": [],
                "context": []
            }
        }


# ==================== 使用示例函数 ====================

def get_response_example():
    """
    使用示例函数 - 需要手动调用
    返回：嫌疑人回答
    """
    # 创建测试数据
    test_case = {
        "truth_events": [
            {
                "eid": "truth_001",
                "type": "bribery",
                "description": "张局长收受李某50万现金贿赂",
                "participants": [{"name": "张局长", "role": "suspect"}, {"name": "李某", "role": "briber"}],
                "is_crime": True
            }
        ],
        "fake_events": [
            {
                "eid": "fake_001",
                "type": "transfer",
                "description": "张局长收到李某50万借款",
                "participants": [{"name": "张局长", "role": "borrower"}, {"name": "李某", "role": "lender"}],
                "deception_type": "distort",
                "narrative": "是李某主动借给我的周转资金"
            }
        ]
    }

    # 创建事件图谱
    event_graph = EventGraph(test_case)

    # 嫌疑人配置
    suspect_profile = {
        "name": "张局长",
        "personality": "Arrogant",
        "speaking_style": "喜欢打官腔"
    }

    # 创建集成器
    integrator = CoTWorkflowIntegrator(event_graph, suspect_profile)

    # 审讯问题
    question = "张局长，1月5号那天下午，你在云隐茶馆见了谁？"

    # 心理状态
    psych_state = {
        "defense_value": 75,
        "stress_value": 30,
        "status_label": "DEFENSIVE"
    }

    # 处理审讯
    result = integrator.process_interrogation_round(question, psych_state)

    return result["response"]