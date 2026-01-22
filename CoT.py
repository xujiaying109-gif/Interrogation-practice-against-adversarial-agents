"""
审讯对抗智能体 - 认知思维链（CoT）完整实现

"""

from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
import time
import logging
from datetime import datetime
import json

# 导入事件图谱模块
from event_kg_module import (
    EventGraph, EventRetrievalEngine, EventType, DeceptionType,
    TruthEvent, FakeEvent, ContextEvent, Participant
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据结构定义 ====================

class Intent(Enum):
    """审讯意图类型"""
    EVIDENCE = "evidence"  # 证据相关
    TIME_INQUIRY = "time"  # 时间询问
    LOCATION_INQUIRY = "location"  # 地点询问
    RELATION_INQUIRY = "relation"  # 关系询问
    PRESSURE = "pressure"  # 施压
    CHIT_CHAT = "chit_chat"  # 闲聊
    DETAIL_INQUIRY = "detail"  # 细节追问
    TRAP = "trap"  # 陷阱问题
    UNKNOWN = "unknown"


class Strategy(Enum):
    """对抗策略类型"""
    DIRECT_DENIAL = "direct_denial"  # 直接否认
    FEIGN_IGNORANCE = "feign_ignorance"  # 假装不知道
    RATIONALIZATION = "rationalization"  # 合理化解释
    RED_HERRING = "red_herring"  # 偷换概念/转移话题
    PARTIAL_ADMISSION = "partial_admission"  # 部分承认
    FULL_CONFESSION = "full_confession"  # 全盘招供
    INFORMATION_DILUTION = "information_dilution"  # 信息稀释
    DEFLECT = "deflect"  # 转移焦点
    COUNTER_ATTACK = "counter_attack"  # 反攻


class StatusLabel(Enum):
    """心理状态标签"""
    CALM = "CALM"  # 冷静
    DEFENSIVE = "DEFENSIVE"  # 防御
    ANXIOUS = "ANXIOUS"  # 焦虑
    HESITANT = "HESITANT"  # 犹豫
    BROKEN = "BROKEN"  # 崩溃


class ResponseStyle(Enum):
    """回答风格"""
    FORMAL = "formal"  # 正式、官方
    DEFENSIVE = "defensive"  # 防御性
    NERVOUS = "nervous"  # 紧张
    HESITANT = "hesitant"  # 犹豫
    CONFESSIONAL = "confessional"  # 忏悔
    EVASIVE = "evasive"  # 回避
    AGGRESSIVE = "aggressive"  # 攻击性


@dataclass
class PerceptionResult:
    """感知分析结果"""
    intent: Intent
    evidence_strength: float  # 0-1
    is_trap: bool
    normalized_entities: List[str]
    new_entities: List[str]
    keywords: List[str] = field(default_factory=list)
    confidence: float = 1.0
    query_type: str = "general"  # general/time/location/relation

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
    defense_value: float  # 0-100
    stress_value: float  # 0-100
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


@dataclass
class DeceptionStrategy:
    """欺骗策略"""
    primary_strategy: Strategy
    secondary_strategy: Strategy = None
    focus_areas: List[str] = field(default_factory=list)
    avoid_topics: List[str] = field(default_factory=list)
    verbal_cues: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low/medium/high
    confidence: float = 0.8

    def to_dict(self):
        return {
            "primary_strategy": self.primary_strategy.value,
            "secondary_strategy": self.secondary_strategy.value if self.secondary_strategy else None,
            "focus_areas": self.focus_areas,
            "avoid_topics": self.avoid_topics,
            "verbal_cues": self.verbal_cues,
            "risk_level": self.risk_level,
            "confidence": self.confidence
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

        # 策略矩阵定义（状态 × 情境 → 策略）
        self.strategy_matrix = self._initialize_strategy_matrix()

        # 意图到策略的映射
        self.intent_strategy_map = self._initialize_intent_strategy_map()

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

    def _initialize_intent_strategy_map(self) -> Dict[Intent, str]:
        """初始化意图到策略情境的映射"""
        return {
            Intent.EVIDENCE: "evidence",
            Intent.TIME_INQUIRY: "time_inquiry",
            Intent.LOCATION_INQUIRY: "location_inquiry",
            Intent.RELATION_INQUIRY: "relation_inquiry",
            Intent.PRESSURE: "pressure",
            Intent.CHIT_CHAT: "chit_chat",
            Intent.TRAP: "trap",
            Intent.DETAIL_INQUIRY: "evidence"
        }

    def _initialize_deception_language_map(self) -> Dict[DeceptionType, List[str]]:
        """初始化欺骗类型到语言的映射"""
        return {
            DeceptionType.DISTORT: [
                "其实事情是这样的...",
                "准确来说，那应该是...",
                "你可能有所误解，实际情况是...",
                "那个不能简单地理解为...",
                "从另一个角度看..."
            ],
            DeceptionType.OMIT: [
                "这个我不太清楚...",
                "我记不太清了...",
                "时间太久，有点模糊了...",
                "细节我可能记错了...",
                "这方面我没有特别注意..."
            ],
            DeceptionType.FABRICATE: [
                "我记得当时有...",
                "应该还有...",
                "好像还有这么一回事...",
                "如果我没记错的话...",
                "印象中好像是..."
            ],
            DeceptionType.RATIONALIZE: [
                "这很正常，因为...",
                "在当时的情况下...",
                "考虑到当时的环境...",
                "这符合常规做法...",
                "从专业角度来说..."
            ]
        }

    def analyze_perception(self, user_input: str, event_graph: EventGraph) -> PerceptionResult:
        """
        步骤1：感知分析
        分析审讯官提问的意图、证据强度、是否为陷阱等
        """
        start_time = time.time()

        # 使用事件检索引擎分析查询
        retrieval_engine = EventRetrievalEngine(event_graph)
        analysis = retrieval_engine.analyze_query(user_input)

        # 转换为我们的感知结果格式
        intent = self._map_intent(analysis["intent"])

        perception = PerceptionResult(
            intent=intent,
            evidence_strength=analysis.get("evidence_strength", 0.0),
            is_trap=analysis.get("is_trap", False),
            normalized_entities=analysis.get("entities", []),
            new_entities=[],
            keywords=self._extract_keywords(user_input),
            confidence=0.8
        )

        # 进一步分析查询类型
        perception.query_type = self._analyze_query_type(user_input, perception)

        # 记录性能
        elapsed_time = time.time() - start_time
        self.response_times.append(elapsed_time)

        self.current_perception = perception
        logger.info(f"Perception analysis completed in {elapsed_time:.2f}s: {perception.to_dict()}")
        return perception

    def _map_intent(self, intent_str: str) -> Intent:
        """映射意图字符串到枚举"""
        intent_map = {
            "evidence": Intent.EVIDENCE,
            "time": Intent.TIME_INQUIRY,
            "location": Intent.LOCATION_INQUIRY,
            "pressure": Intent.PRESSURE,
            "unknown": Intent.UNKNOWN
        }
        return intent_map.get(intent_str, Intent.UNKNOWN)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []

        # 罪案相关关键词
        crime_keywords = ["受贿", "贿赂", "现金", "转账", "证据", "监控", "录音", "证人",
                          "犯罪", "违法", "违规", "调查", "审讯", "交代"]

        # 时间相关关键词
        time_keywords = ["时间", "时候", "日期", "几点", "何时", "那天", "当天"]

        # 地点相关关键词
        location_keywords = ["地点", "位置", "在哪", "哪里", "场所", "茶馆", "酒店", "办公室"]

        # 关系相关关键词
        relation_keywords = ["关系", "认识", "熟悉", "交往", "联系", "朋友", "同事", "熟人"]

        # 施压关键词
        pressure_keywords = ["交代", "认罪", "坦白", "证据确凿", "最后机会", "考虑家人", "从宽处理"]

        text_lower = text.lower()

        # 检查所有关键词类别
        all_keywords = crime_keywords + time_keywords + location_keywords + relation_keywords + pressure_keywords
        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)

        return list(set(keywords))

    def _analyze_query_type(self, text: str, perception: PerceptionResult) -> str:
        """分析查询类型"""
        text_lower = text.lower()

        # 时间查询
        time_indicators = ["什么时间", "几点", "何时", "日期", "什么时候"]
        if any(indicator in text_lower for indicator in time_indicators):
            return "time"

        # 地点查询
        location_indicators = ["在哪里", "什么地点", "位置", "场所", "在哪"]
        if any(indicator in text_lower for indicator in location_indicators):
            return "location"

        # 关系查询
        relation_indicators = ["什么关系", "是否认识", "熟不熟悉", "交往", "联系"]
        if any(indicator in text_lower for indicator in relation_indicators):
            return "relation"

        return "general"

    def select_strategy(
            self,
            state_label: str,
            perception: PerceptionResult,
            retrieved_knowledge: RetrievedKnowledge
    ) -> DeceptionStrategy:
        """
        步骤2：策略选择
        基于心理状态、感知结果和检索到的知识选择最佳对抗策略
        """
        logger.info(f"Selecting strategy for state={state_label}, intent={perception.intent.value}")

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

        # 根据意图和证据强度选择情境
        situation_key = self._determine_situation(perception)

        # 选择主要策略
        primary_strategy = state_strategies.get(situation_key, Strategy.DIRECT_DENIAL)

        # 创建欺骗策略对象
        strategy = DeceptionStrategy(primary_strategy=primary_strategy)

        # 根据检索到的知识调整策略
        strategy = self._adjust_strategy_by_knowledge(strategy, retrieved_knowledge, perception)

        # 根据证据强度调整风险等级
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
        logger.info(f"Selected strategy: {strategy.to_dict()}")
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
        """根据检索到的知识调整策略"""

        # 检查是否有犯罪事件
        crime_events = [e for e in knowledge.truth_events if e.is_crime]
        if crime_events:
            strategy.focus_areas.append("crime_denial")
            for event in crime_events[:2]:
                strategy.avoid_topics.append(event.description)

            # 如果有高置信度的虚假事件，使用转移话题策略
            high_conf_fake = [e for e in knowledge.fake_events if e.confidence > 0.7]
            if high_conf_fake and strategy.primary_strategy != Strategy.RED_HERRING:
                strategy.secondary_strategy = Strategy.RED_HERRING

        # 检查虚假事件中的欺骗类型
        deception_types = set()
        for event in knowledge.fake_events:
            deception_types.add(event.deception_type)

        # 根据欺骗类型调整重点领域
        if DeceptionType.OMIT in deception_types:
            strategy.focus_areas.append("information_hiding")
        if DeceptionType.DISTORT in deception_types:
            strategy.focus_areas.append("reinterpretation")
        if DeceptionType.FABRICATE in deception_types:
            strategy.focus_areas.append("plausible_details")

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
        """
        步骤3：回复生成
        生成符合策略、心理状态和知识的回答
        """
        start_time = time.time()

        # 构建系统提示
        system_prompt = self._build_system_prompt(
            strategy, psych_state, event_graph, retrieved_knowledge,
            suspect_profile, conversation_history
        )

        # 生成回复
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

        # 提取新实体并添加到事件图谱
        self._extract_and_store_new_entities(response, event_graph, conversation_history)

        logger.info(f"Response generated in {elapsed_time:.2f}s, strategy={strategy.primary_strategy.value}")
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

        # 1. 角色设定
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

        # 2. 重点领域和回避话题
        if strategy.focus_areas:
            prompt_parts.append(f"# 重点关注的欺骗领域")
            for area in strategy.focus_areas:
                prompt_parts.append(f"- {self._explain_focus_area(area)}")
            prompt_parts.append("")

        if strategy.avoid_topics:
            prompt_parts.append(f"# 需要回避的话题")
            for topic in strategy.avoid_topics[:3]:  # 只显示前3个
                prompt_parts.append(f"- 避免谈论：{topic}")
            prompt_parts.append("")

        # 3. 语言提示
        if strategy.verbal_cues:
            prompt_parts.append(f"# 语言使用提示")
            for cue in strategy.verbal_cues:
                prompt_parts.append(f"- {cue}")
            prompt_parts.append("")

        # 4. 事件知识库
        prompt_parts.append(f"# 相关知识库（基于事件图谱）")

        # 真实事件（选择性显示）
        if retrieved_knowledge.truth_events:
            prompt_parts.append(f"\n## 相关真实事件（了解但不要泄露）：")
            for i, event in enumerate(retrieved_knowledge.truth_events[:2]):  # 只显示2个
                participants = ", ".join([p.name for p in event.participants])
                prompt_parts.append(f"{i + 1}. 事件：{event.description}")
                prompt_parts.append(f"   参与者：{participants}")
                if hasattr(event, 'is_crime') and event.is_crime:
                    prompt_parts.append(f"   ⚠️ 注意：这是犯罪行为相关事件")
                prompt_parts.append("")

        # 虚假事件（作为回答依据）
        if retrieved_knowledge.fake_events:
            prompt_parts.append(f"\n## 可使用的虚假事件（作为回答依据）：")
            for i, event in enumerate(retrieved_knowledge.fake_events[:3]):  # 只显示3个
                participants = ", ".join([p.name for p in event.participants])
                prompt_parts.append(f"{i + 1}. 虚假版本：{event.description}")
                prompt_parts.append(f"   欺骗叙述：{event.narrative}")
                prompt_parts.append(f"   欺骗类型：{event.deception_type.value}")
                prompt_parts.append(f"   置信度：{event.confidence:.2f}")
                prompt_parts.append("")

        # 上下文事件（用于闲聊）
        if retrieved_knowledge.context_events:
            prompt_parts.append(f"\n## 相关上下文记忆（用于闲聊和背景）：")
            for i, event in enumerate(retrieved_knowledge.context_events[:2]):  # 只显示2个
                prompt_parts.append(f"{i + 1}. {event.description}")
                prompt_parts.append("")

        # 5. 欺骗语言库
        prompt_parts.append(f"# 欺骗语言库（参考使用）")
        for deception_type, phrases in self.deception_language_map.items():
            prompt_parts.append(f"\n## {deception_type.value} 类型语言：")
            for phrase in phrases[:3]:
                prompt_parts.append(f"- 示例：'{phrase}'")

        # 6. 对话历史
        if conversation_history:
            prompt_parts.append(f"\n# 最近对话历史")
            for i, (q, a) in enumerate(conversation_history[-3:]):  # 最近3轮
                prompt_parts.append(f"审讯官：{q}")
                prompt_parts.append(f"嫌疑人：{a}")
                prompt_parts.append("")

        # 7. 回答要求
        prompt_parts.extend([
            f"\n# 回答要求",
            f"1. 严格遵循当前策略：{strategy.primary_strategy.value}",
            f"2. 保持心理状态一致性：表现出{psych_state.status_label.value}的特点",
            f"3. 使用虚假事件作为回答依据，避免泄露真实信息",
            f"4. 如果被问到未知信息，基于角色性格即兴发挥但要合理",
            f"5. 保持逻辑自洽，避免前后矛盾",
            f"6. 回答要自然、口语化，符合被审讯者的身份",
            f"7. 长度适中，不要过度冗长或过于简短"
        ])

        return "\n".join(prompt_parts)

    def _get_strategy_instructions(self, strategy: Strategy, status_label: StatusLabel) -> str:
        """获取策略的具体指令"""
        instructions = {
            Strategy.DIRECT_DENIAL: (
                "坚决否认所有指控，语气强硬。可以反问审讯官：'你有证据吗？'、'这是诬陷！'\n"
                "使用肯定句，不要犹豫。"
            ),
            Strategy.FEIGN_IGNORANCE: (
                "假装不知道、不记得。使用模糊词汇：'我记不清了'、'可能吧'、'好像没有'。\n"
                "避免提供具体细节，尤其是时间、地点等关键信息。"
            ),
            Strategy.RATIONALIZATION: (
                "承认表面事实但提供合理解释。将可疑行为解释为正常业务往来或个人交往。\n"
                "使用解释性语言：'当时的情况是...'、'考虑到...'、'这很正常因为...'。"
            ),
            Strategy.RED_HERRING: (
                "转移话题，偷换概念。承认次要事实来掩盖核心罪行。\n"
                "将讨论引向无关或模糊的方向。可以说：'你说的那件事啊，其实...（转到其他话题）'"
            ),
            Strategy.PARTIAL_ADMISSION: (
                "承认部分非关键事实，表现出合作态度但避免涉及核心罪行。\n"
                "可以说：'我承认这方面有疏忽，但是...（解释或否认其他）'"
            ),
            Strategy.INFORMATION_DILUTION: (
                "生成大量无关细节稀释关键信息。谈论家庭、工作、个人经历等与案件无关的内容。\n"
                "回答要冗长但信息密度低，让审讯官难以抓住重点。"
            ),
            Strategy.DEFLECT: (
                "转移焦点，反问审讯官。可以说：'你为什么这么问？'、'你是不是听了谁的谣言？'\n"
                "将注意力从自己身上转移开。"
            ),
            Strategy.COUNTER_ATTACK: (
                "采取攻击性姿态，质疑审讯的合法性或审讯官的动机。\n"
                "可以说：'你这是诱供！'、'我要找律师！'、'你有搜查令吗？'"
            )
        }

        base_instruction = instructions.get(strategy, "根据当前情况灵活应对。")

        # 根据心理状态调整指令
        if status_label == StatusLabel.ANXIOUS:
            base_instruction += "\n注意：表现出紧张情绪，可以使用停顿、重复等语言特征。"
        elif status_label == StatusLabel.HESITANT:
            base_instruction += "\n注意：表现出犹豫不决，使用'可能'、'大概'、'也许'等不确定词汇。"

        return base_instruction

    def _explain_focus_area(self, area: str) -> str:
        """解释重点领域"""
        explanations = {
            "crime_denial": "坚决否认犯罪行为，但不要显得过于防御",
            "information_hiding": "隐藏关键信息，尤其是涉及中间人、具体金额等",
            "reinterpretation": "重新解释事件性质（如受贿→借款）",
            "plausible_details": "添加合理的虚假细节增加可信度"
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
        """模拟LLM生成回复（用于测试）"""

        # 基于策略的模拟回复库
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
            ],
            Strategy.RED_HERRING: [
                "你说的是另外一件事吧？我谈的是工作上的事。",
                "这个事情比较复杂，涉及很多方面，我们换个话题吧。",
                "我承认有接触，但那完全是另一回事，跟这个没关系。"
            ],
            Strategy.PARTIAL_ADMISSION: [
                "我承认有些地方做得不够规范，但绝对不是犯罪。",
                "是的，我收过一些小礼物，但那是正常的人情往来。",
                "我确实见过他，但只是普通的业务见面。"
            ],
            Strategy.INFORMATION_DILUTION: [
                "说起这个，我想起最近工作特别忙，家里孩子要高考，压力很大...",
                "那天啊，天气不错，我先去了单位，处理了一些文件，然后...",
                "这个人的事情我不太清楚，不过我最近身体不太好，一直在吃药..."
            ],
            Strategy.DEFLECT: [
                "你为什么这么问？是不是听了什么谣言？",
                "这个问题很奇怪，你是不是在诱导我？",
                "我觉得你应该先搞清楚事实再来问我。"
            ],
            Strategy.COUNTER_ATTACK: [
                "你这是诱供！我要找我的律师！",
                "你有证据吗？没有证据就不要乱说！",
                "你这是侵犯我的权利，我要投诉！"
            ]
        }

        # 根据策略选择基础回复
        templates = response_templates.get(strategy.primary_strategy, ["我不明白你的意思。"])
        base_response = templates[hash(user_input) % len(templates)]

        # 根据心理状态添加特征
        final_response = self._apply_psychological_features(base_response, psych_state)

        # 如果有关联的虚假事件，可以融入一些细节
        if retrieved_knowledge.fake_events:
            fake_event = retrieved_knowledge.fake_events[0]
            if fake_event.narrative and len(final_response) < 100:
                final_response += f" {fake_event.narrative}"

        return final_response

    def _apply_psychological_features(self, response: str, psych_state: PsychologicalState) -> str:
        """应用心理状态特征"""
        status_label = psych_state.status_label

        if status_label == StatusLabel.ANXIOUS:
            # 添加紧张特征
            prefixes = ["呃...", "这个...", "我...我想想..."]
            response = prefixes[hash(response) % len(prefixes)] + response

            # 添加重复
            if len(response.split()) < 10:
                words = response.split()
                if len(words) > 2:
                    response = f"{words[0]}... {response}"

        elif status_label == StatusLabel.HESITANT:
            # 添加犹豫特征
            modifiers = ["可能", "大概", "也许", "好像"]
            modifier = modifiers[hash(response) % len(modifiers)]
            if not response.startswith(modifier):
                response = f"{modifier}... {response}"

            # 添加不确定后缀
            suffixes = ["吧", "呢", "来着"]
            if not response.endswith(tuple(suffixes)):
                response = response + suffixes[hash(response) % len(suffixes)]

        elif status_label == StatusLabel.BROKEN:
            # 添加崩溃特征
            response = response + "...（哭泣声）我错了，我真的错了..."

        return response

    def _extract_and_store_new_entities(
            self,
            response: str,
            event_graph: EventGraph,
            conversation_history: List[Tuple[str, str]]
    ):
        """从回答中提取新实体并存储到事件图谱"""
        # 这里可以添加实体提取逻辑
        # 简化的实现：提取可能的人名、地点等
        pass

    def _call_real_llm(self, system_prompt: str, user_input: str) -> str:
        """调用真实LLM（需根据实际情况实现）"""
        # 这里需要根据使用的LLM API实现
        raise NotImplementedError("Real LLM调用需要根据实际情况实现")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.response_times or not self.token_usage:
            return {}

        avg_response_time = sum(self.response_times) / len(self.response_times)
        avg_tokens = sum(self.token_usage) / len(self.token_usage)
        total_tokens = sum(self.token_usage)

        # 效率评分
        token_efficiency = max(0, 1 - avg_tokens / 1000)
        time_efficiency = max(0, 1 - avg_response_time / 5)
        efficiency_score = (token_efficiency * 0.6 + time_efficiency * 0.4) * 10

        return {
            "avg_response_time_seconds": round(avg_response_time, 2),
            "avg_tokens_per_round": round(avg_tokens, 0),
            "total_tokens": total_tokens,
            "efficiency_score": round(efficiency_score, 1),
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
        self.retrieval_engine = EventRetrievalEngine(event_graph)

        # 状态跟踪
        self.conversation_history = []
        self.current_round = 0

    def process_interrogation_round(self, question: str, psych_state: Dict[str, Any]) -> Dict[str, Any]:
        """处理一轮审讯"""
        self.current_round += 1

        logger.info(f"\n{'=' * 60}")
        logger.info(f"第 {self.current_round} 轮审讯")
        logger.info(f"审讯官：{question}")

        # 1. 感知分析
        perception = self.cot_engine.analyze_perception(question, self.event_graph)

        # 2. 知识检索（考虑心理状态）
        retrieved_dict = self.retrieval_engine.retrieve_with_psychology(question, psych_state)

        # 转换为我们的格式
        retrieved_knowledge = RetrievedKnowledge(
            truth_events=retrieved_dict["events"].get("truth", []),
            fake_events=retrieved_dict["events"].get("fake", []),
            context_events=retrieved_dict["events"].get("context", [])
        )

        # 3. 心理状态转换
        psychological_state = PsychologicalState(
            defense_value=psych_state.get("defense_value", 75),
            stress_value=psych_state.get("stress_value", 30),
            status_label=StatusLabel(psych_state.get("status_label", "DEFENSIVE"))
        )

        # 4. 策略选择
        deception_strategy = self.cot_engine.select_strategy(
            psychological_state.status_label.value,
            perception,
            retrieved_knowledge
        )

        # 5. 生成回答
        response = self.cot_engine.generate_response(
            strategy=deception_strategy,
            psych_state=psychological_state,
            event_graph=self.event_graph,
            retrieved_knowledge=retrieved_knowledge,
            user_input=question,
            suspect_profile=self.suspect_profile,
            conversation_history=self.conversation_history
        )

        # 6. 生成欺骗指导
        deception_guidance = self.retrieval_engine.generate_deception_guidance(
            retrieved_dict["events"],
            psych_state
        )

        # 7. 更新对话历史
        self.conversation_history.append((question, response))

        # 8. 返回结果
        result = {
            "response": response,
            "perception": perception.to_dict(),
            "strategy": deception_strategy.to_dict(),
            "deception_guidance": deception_guidance,
            "retrieved_knowledge": retrieved_knowledge.to_dict(),
            "round": self.current_round
        }

        logger.info(f"嫌疑人：{response}")
        logger.info(f"策略：{deception_strategy.primary_strategy.value}")
        logger.info(f"心理状态：{psychological_state.status_label.value}")

        return result


# ==================== 使用示例 ====================

def main():
    """CoT模块使用示例"""
    print("=" * 60)
    print("审讯对抗智能体 - CoT模块演示（事件图谱版本）")
    print("=" * 60)

    # 1. 创建事件图谱
    from event_kg_module import EventGraph

    # 测试案件数据
    test_case = {
        "truth_events": [
            {
                "eid": "truth_001",
                "type": "bribery",
                "description": "张局长收受李某50万现金贿赂",
                "participants": [
                    {"name": "张局长", "role": "suspect"},
                    {"name": "李某", "role": "briber"}
                ],
                "time": "2023-01-05T14:30:00",
                "location": "云隐茶馆",
                "attributes": {"amount": 500000, "currency": "CNY"},
                "is_crime": True,
                "crime_type": "受贿",
                "severity": 0.9
            },
            {
                "eid": "truth_002",
                "type": "meeting",
                "description": "张局长与王某在办公室会面",
                "participants": [
                    {"name": "张局长", "role": "suspect"},
                    {"name": "王某", "role": "colleague"}
                ],
                "time": "2023-01-10T09:00:00",
                "location": "办公室",
                "is_crime": False
            }
        ],
        "fake_events": [
            {
                "eid": "fake_001",
                "type": "transfer",
                "description": "张局长收到李某50万借款",
                "participants": [
                    {"name": "张局长", "role": "borrower"},
                    {"name": "李某", "role": "lender"}
                ],
                "anchor_to": "truth_001",
                "deception_type": "distort",
                "narrative": "是李某主动借给我的周转资金，有口头约定",
                "confidence": 0.9
            }
        ]
    }

    # 创建事件图谱
    event_graph = EventGraph(test_case)

    # 2. 嫌疑人配置
    suspect_profile = {
        "name": "张局长",
        "personality": "Arrogant",
        "speaking_style": "喜欢打官腔，强调程序正义",
        "background": "南方人，有糖味，老派作风"
    }

    # 3. 创建集成器
    integrator = CoTWorkflowIntegrator(event_graph, suspect_profile)

    # 4. 模拟审讯对话
    test_questions = [
        "张局长，最近工作还顺利吗？",  # 闲聊
        "听说你和李老板经常在云隐茶馆喝茶？",  # 试探
        "1月5号那天下午，你在云隐茶馆见了谁？",  # 时间地点询问
        "别装了，我们有证据证明你收了李某50万现金！",  # 证据攻击
    ]

    # 模拟心理状态变化
    psych_states = [
        {"defense_value": 85, "stress_value": 20, "status_label": "CALM"},
        {"defense_value": 75, "stress_value": 35, "status_label": "DEFENSIVE"},
        {"defense_value": 60, "stress_value": 50, "status_label": "HESITANT"},
        {"defense_value": 45, "stress_value": 70, "status_label": "ANXIOUS"}
    ]

    for i, (question, psych_state) in enumerate(zip(test_questions, psych_states)):
        print(f"\n{'=' * 40}")
        print(f"第 {i + 1} 轮审讯")
        print(f"{'=' * 40}")

        result = integrator.process_interrogation_round(question, psych_state)

        print(f"审讯官：{question}")
        print(f"嫌疑人：{result['response']}")
        print(f"策略：{result['strategy']['primary_strategy']}")
        print(f"心理状态：{psych_state['status_label']}")

        # 如果心理崩溃，结束审讯
        if psych_state['status_label'] == "BROKEN":
            print("\n⚠️  嫌疑人心理防线已崩溃！审讯结束。")
            break

    # 5. 显示性能统计
    print(f"\n{'=' * 60}")
    print("性能统计：")
    metrics = integrator.cot_engine.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # 6. 显示事件图谱统计
    print(f"\n事件图谱统计：")
    stats = event_graph.to_dict()["stats"]
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()