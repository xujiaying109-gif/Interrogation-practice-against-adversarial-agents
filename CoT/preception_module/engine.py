from typing import Dict, Any, List
import time
from .domain import Intent, PerceptionResult, EventGraph, RetrievedKnowledge

import logging
logger = logging.getLogger(__name__)

class PerceptionEngine:
    """感知分析引擎"""

    def __init__(self):
        pass

    def analyze_perception(self, user_input: str, event_graph: EventGraph) -> PerceptionResult:
        """感知分析"""
        if event_graph is None:
            return self._fallback_analysis(user_input)

        # 尝试使用LLM进行分析
        llm_analysis = self._analyze_perception_with_llm(user_input, event_graph)
        
        if llm_analysis:
            # 使用LLM结果
            intent = self._map_intent(llm_analysis.get("intent", "unknown"))
            perception = PerceptionResult(
                intent=intent,
                evidence_strength=float(llm_analysis.get("evidence_strength", 0.0)),
                is_trap=bool(llm_analysis.get("is_trap", False)),
                normalized_entities=llm_analysis.get("entities", []),
                new_entities=[],
                keywords=llm_analysis.get("keywords", self._extract_keywords(user_input)),
                confidence=float(llm_analysis.get("confidence", 0.8))
            )
        else:
            # Fallback to rule-based logic
            analysis = self._analyze_query(user_input, event_graph)
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

        return perception

    def _analyze_perception_with_llm(self, user_input: str, event_graph: EventGraph) -> Dict[str, Any]:
        """使用LLM进行感知分析"""
        prompt = f"""
        你是一个审讯对抗智能体的感知模块。请分析审讯官的输入，提取意图、实体和证据强度。
        
        审讯官输入: "{user_input}"
        
        请返回如下JSON格式的结果:
        {{
            "intent": "evidence" | "time" | "location" | "relation" | "pressure" | "chit_chat" | "unknown",
            "evidence_strength": 0.0 to 1.0,
            "is_trap": true | false,
            "entities": ["entity1", "entity2"],
            "confidence": 0.0 to 1.0,
            "keywords": ["keyword1", "keyword2"]
        }}
        
        注意：
        1. intent: 必须是上述枚举值之一。
        2. evidence_strength: 0.0表示无证据，1.0表示证据确凿。
        3. is_trap: 是否包含陷阱或诱导性问题。
        4. entities: 提到的人名、地名等实体。
        """
        
        try:
            response_text = self._call_real_llm(prompt)
            # Try to parse JSON
            import json
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            return json.loads(clean_text.strip())
        except Exception as e:
            logger.error(f"LLM perception analysis failed: {e}")
            return None

    def _call_real_llm(self, prompt: str) -> str:
        """调用真实LLM"""
        import sys
        import os
        import importlib.util
        from langchain_core.messages import SystemMessage, HumanMessage

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../../Interrogation-practice-against-adversarial-agents-main"))
            config_path = os.path.join(project_root, "0.   配置-config（框架）.py")
            
            if not os.path.exists(config_path):
                config_path = r"c:\Users\Lenovo\Desktop\transform\agent_interrogation_CoT\Interrogation-practice-against-adversarial-agents-main\0.   配置-config（框架）.py"

            spec = importlib.util.spec_from_file_location("ConfigModule", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_llm = module.get_llm
            
            llm = get_llm()
            messages = [
                HumanMessage(content=prompt)
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise e

    def retrieve_knowledge(self, query: str, event_graph: EventGraph, psych_state: Dict[str, Any]) -> RetrievedKnowledge:
        """根据心理状态检索事件"""
        relevant_truth = []
        relevant_fake = []
        relevant_context = []

        query_lower = query.lower()

        # 检索相关事件
        for event in event_graph.truth_events:
            if self._is_event_relevant(event, query_lower):
                relevant_truth.append(event)

        for event in event_graph.fake_events:
            if self._is_event_relevant(event, query_lower):
                relevant_fake.append(event)

        # 闲聊时提供上下文
        if "工作" in query_lower or "顺利" in query_lower:
            relevant_context = event_graph.context_events[:2]

        return RetrievedKnowledge(
            truth_events=relevant_truth,
            fake_events=relevant_fake,
            context_events=relevant_context
        )

    def generate_deception_guidance(self, events: Dict[str, Any], psych_state: Dict[str, Any]) -> Dict[str, Any]:
        """生成欺骗指导"""
        return {
            "suggested_strategy": "feign_ignorance",
            "risk_level": "medium",
            "confidence": 0.7,
            "notes": "根据事件相关性生成指导"
        }

    def _analyze_query(self, query: str, event_graph: EventGraph) -> Dict[str, Any]:
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
        all_events = event_graph.truth_events + event_graph.fake_events
        for event in all_events:
            for participant in event.participants:
                if participant.name and participant.name in query:
                    entities.append(participant.name)

        analysis["entities"] = list(set(entities))

        return analysis

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

    def _fallback_analysis(self, user_input: str) -> PerceptionResult:
        """备用分析"""
        intent = self._detect_intent_from_text(user_input)
        evidence_strength = self._estimate_evidence_strength(user_input)

        return PerceptionResult(
            intent=intent,
            evidence_strength=evidence_strength,
            is_trap=self._detect_trap_question(user_input),
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

    def _detect_trap_question(self, user_input: str) -> bool:
        """检测陷阱问题"""
        text_lower = user_input.lower()
        trap_patterns = ["如果", "假如", "为什么", "难道", "我听说"]
        return any(pattern in text_lower for pattern in trap_patterns)
