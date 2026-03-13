"""
感知模块 - 大模型语义意图与实体分析
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import json
import prompts
from log_utils import get_logger

logger = get_logger(__name__)


class Intent(Enum):
    EVIDENCE = "evidence"
    TIME_INQUIRY = "time_inquiry"
    LOCATION_INQUIRY = "location_inquiry"
    RELATION_INQUIRY = "relation_inquiry"
    PRESSURE = "pressure"
    TRAP = "trap"
    UNKNOWN = "unknown"


@dataclass
class PerceptionResult:
    intent: Intent
    evidence_strength: float
    is_trap: bool
    keywords: List[str]


class PerceptionModule:
    def __init__(self, api_client=None):
        self.api_client = api_client

    def analyze(self, question: str) -> PerceptionResult:
        # 如果没有 API 客户端，启用降级字面逻辑
        if not self.api_client:
            return self._fallback_analyze(question)

        system_prompt = prompts.SYSTEM_PROMPT_PERCEPTION
        user_input = f"【主审官提问】: {question}"

        try:
            raw_res = self.api_client.generate_response(
                system_prompt=system_prompt,
                user_input=user_input,
                temperature=0.1,
                max_tokens=200
            ).strip()

            if raw_res.startswith("```json"):
                raw_res = raw_res[7:]
            if raw_res.startswith("```"):
                raw_res = raw_res[3:]
            if raw_res.endswith("```"):
                raw_res = raw_res[:-3]
            raw_res = raw_res.strip()

            parsed = json.loads(raw_res)

            intent_str = parsed.get("intent", "unknown")
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN

            strength = float(parsed.get("evidence_strength", 0.2))
            is_trap = bool(parsed.get("is_trap", False))
            keywords = parsed.get("keywords", [])

            return PerceptionResult(intent, strength, is_trap, keywords)

        except Exception as e:
            logger.error(f"Perception LLM failed, using fallback. Error: {e}")
            return self._fallback_analyze(question)

    def _fallback_analyze(self, question: str) -> PerceptionResult:
        q = question.lower()
        intent = Intent.UNKNOWN
        strength = 0.2
        is_trap = False
        keywords = []

        if any(w in q for w in ["证据", "监控", "录音", "账单", "流水"]):
            intent = Intent.EVIDENCE
            strength = 0.8
            keywords.extend(["证据", "账单"])
        elif any(w in q for w in ["交代", "坦白", "认罪", "招供"]):
            intent = Intent.PRESSURE
            strength = 0.6
        elif any(w in q for w in ["时间", "几点", "哪天", "何时"]):
            intent = Intent.TIME_INQUIRY
            keywords.append("时间")
        elif any(w in q for w in ["哪里", "地点", "位置", "在哪"]):
            intent = Intent.LOCATION_INQUIRY
            keywords.append("地点")

        if any(w in q for w in ["如果", "假设", "听说", "真的吗", "确定吗"]):
            is_trap = True

        return PerceptionResult(intent, strength, is_trap, keywords)