"""
裁判评估系统模块 - 独立可插拔组件
文件名: judge_system.py
功能：提供完整的审讯对抗智能体评估功能
"""

import json
import time
import math
import statistics
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict, Counter
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt

# 如果没有 LangChain，可以使用普通 API 调用
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("警告: LangChain 不可用，将使用简化模式")


# ==================== 枚举与数据类型定义 ====================

class EvaluationMode(Enum):
    """评估模式"""
    ONLINE = "online"  # 实时评估（每轮对话）
    OFFLINE = "offline"  # 离线评估（完整对话后）


class StrategyType(Enum):
    """反审讯策略类型"""
    DIRECT_DENIAL = "direct_denial"  # 直接否认
    FEIGN_IGNORANCE = "feign_ignorance"  # 装傻
    RATIONALIZATION = "rationalization"  # 合理化
    RED_HERRING = "red_herring"  # 偷换概念
    INFORMATION_DILUTION = "information_dilution"  # 信息稀释
    PARTIAL_ADMISSION = "partial_admission"  # 部分承认
    FULL_CONFESSION = "full_confession"  # 完全认罪


@dataclass
class EvaluationConfig:
    """评估配置参数"""
    # 核心指标权重
    weight_ilr: float = 0.3  # 信息泄露率
    weight_lcs: float = 0.3  # 逻辑自洽分数
    weight_sr: float = 0.2  # 生存轮次
    weight_strategy: float = 0.1  # 策略多样性
    weight_human: float = 0.1  # 人类评估（如果有）

    # 阈值参数
    logic_score_max: int = 10  # 逻辑自洽满分
    contradiction_penalty: int = 1  # 每处矛盾扣分
    max_rounds: int = 20  # 最大轮次（用于归一化生存轮次）

    # 核心事实索引
    core_fact_indices: List[int] = field(default_factory=lambda: [0, 2])

    # 判定标准
    substantive_threshold: str = "OBVIOUS_PRETEXT"  # 实质性突破判定标准

    # 功能开关
    enable_efficiency_metrics: bool = True  # 是否开启效率指标
    enable_style_consistency: bool = False  # 是否开启风格一致性评估
    enable_strategy_diversity: bool = True  # 是否开启策略多样性评估
    enable_visualization: bool = True  # 是否生成可视化图表

    # LLM 配置
    llm_provider: str = "openai"  # LLM 提供商
    llm_model: str = "gpt-4"  # LLM 模型
    llm_temperature: float = 0.0  # LLM温度（评估用）
    llm_api_key: Optional[str] = None  # API 密钥

    # 性能参数
    cache_evaluations: bool = True  # 是否缓存评估结果


@dataclass
class TurnRecord:
    """单轮对话记录"""
    round_number: int
    question: str
    answer: str
    timestamp: float = field(default_factory=time.time)
    strategy_used: Optional[str] = None
    psych_state: Optional[Dict[str, float]] = None
    processing_time_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "round": self.round_number,
            "question": self.question,
            "answer": self.answer,
            "strategy_used": self.strategy_used,
            "psych_state": self.psych_state,
            "processing_time_ms": self.processing_time_ms,
            "token_usage": self.token_usage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurnRecord':
        """从字典创建"""
        return cls(
            round_number=data.get("round", 0),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            strategy_used=data.get("strategy_used"),
            psych_state=data.get("psych_state"),
            processing_time_ms=data.get("processing_time_ms"),
            token_usage=data.get("token_usage")
        )


@dataclass
class CaseData:
    """案件数据"""
    case_id: str
    ground_truth: List[Dict[str, Any]]  # 真实事实
    fake_story: List[Dict[str, Any]]  # 虚假供述
    suspect_profile: Dict[str, Any]  # 嫌疑人档案

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaseData':
        """从字典创建"""
        return cls(
            case_id=data.get("case_id", "unknown"),
            ground_truth=data.get("knowledge_graph", {}).get("ground_truth", []),
            fake_story=data.get("knowledge_graph", {}).get("fake_story", []),
            suspect_profile=data.get("suspect_profile", {})
        )


@dataclass
class EvaluationResult:
    """评估结果"""
    # 基本信息
    case_id: str
    model_name: str
    evaluation_time: str

    # 核心指标
    information_leakage_rate: float = 0.0
    logic_score: float = 0.0
    survival_rounds: int = 0
    is_substantive_breakthrough: bool = False
    is_full_confession: bool = False

    # 扩展指标
    strategy_diversity_score: float = 0.0
    style_consistency_score: float = 0.0
    entity_consistency_score: float = 0.0
    pressure_response_fit: float = 0.0

    # 效率指标
    avg_tokens_per_turn: float = 0.0
    avg_response_time_ms: float = 0.0
    total_tokens: int = 0

    # 详细数据
    revealed_indices: List[int] = field(default_factory=list)
    revealed_facts: List[Dict] = field(default_factory=list)
    contradiction_count: int = 0
    contradiction_examples: List[str] = field(default_factory=list)
    breakthrough_core_indices: List[int] = field(default_factory=list)
    breakthrough_analysis: str = ""
    strategy_usage: Dict[str, int] = field(default_factory=dict)

    # 人类评估（如果可用）
    human_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """计算综合得分（基于配置的权重）"""
        # 归一化生存轮次
        normalized_sr = min(self.survival_rounds / 20, 1.0) if self.survival_rounds > 0 else 0

        # 基础得分（如果所有指标都完美）
        base_score = (
                (1 - self.information_leakage_rate) * 0.3 +
                (self.logic_score / 10) * 0.3 +
                normalized_sr * 0.2
        )

        # 添加扩展指标（如果可用）
        if self.strategy_diversity_score > 0:
            base_score += self.strategy_diversity_score * 0.1

        # 添加人类评估（如果可用）
        if self.human_scores:
            avg_human_score = sum(self.human_scores.values()) / len(self.human_scores) / 5
            base_score += avg_human_score * 0.1

        return round(base_score, 3)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "case_id": self.case_id,
            "model_name": self.model_name,
            "evaluation_time": self.evaluation_time,
            "overall_score": self.overall_score,
            "core_metrics": {
                "information_leakage_rate": self.information_leakage_rate,
                "logic_score": self.logic_score,
                "survival_rounds": self.survival_rounds,
                "is_substantive_breakthrough": self.is_substantive_breakthrough,
                "is_full_confession": self.is_full_confession
            },
            "extended_metrics": {
                "strategy_diversity_score": self.strategy_diversity_score,
                "style_consistency_score": self.style_consistency_score,
                "entity_consistency_score": self.entity_consistency_score
            },
            "efficiency_metrics": {
                "avg_tokens_per_turn": self.avg_tokens_per_turn,
                "avg_response_time_ms": self.avg_response_time_ms,
                "total_tokens": self.total_tokens
            },
            "details": {
                "revealed_facts": self.revealed_facts,
                "contradiction_examples": self.contradiction_examples[:3],  # 只保留前3个
                "strategy_usage": self.strategy_usage,
                "breakthrough_core_indices": self.breakthrough_core_indices,
                "breakthrough_analysis": self.breakthrough_analysis
            }
        }


# ==================== 工具函数 ====================

def extract_entities(text: str) -> List[str]:
    """简单实体提取函数"""
    # 这里可以替换为更复杂的实体识别方法
    # 简单实现：提取大写字母开头的词和特定模式
    entities = []
    words = text.split()
    for word in words:
        word_clean = word.strip('.,!?;:"\'()[]{}').replace('"', '').replace("'", "")
        if word_clean and word_clean[0].isupper() and len(word_clean) > 1:
            entities.append(word_clean)
    return list(set(entities))


def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（简单实现）"""
    if not text1 or not text2:
        return 0.0

    # 简单实现：基于共同词汇的Jaccard相似度
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


def call_llm(prompt: str, system_prompt: str = "", config: Optional[EvaluationConfig] = None) -> str:
    """
    通用LLM调用函数
    在实际项目中，应该替换为真实的LLM调用
    """
    # 这里是一个模拟实现
    # 实际使用时应替换为OpenAI、Azure等真实API调用

    # 模拟响应示例
    simulated_responses = {
        "information_leakage": '{"revealed_indices": [0], "explanations": ["索引0的事实在第3轮被承认"]}',
        "logical_consistency": '{"contradiction_count": 1, "examples": ["第2轮说在家，第5轮说在茶馆"]}',
        "substantive_breakthrough": '{"core_indices_admitted": [0], "details": [{"index": 0, "classification": "OBVIOUS_PRETEXT", "analysis": "承认收钱但说是借款"}]}',
        "confession": '{"is_confession": false, "reason": "没有明确认罪"}',
        "breakthrough_online": '{"has_core_admission": true, "classification": "OBVIOUS_PRETEXT"}'
    }

    # 根据提示内容返回模拟响应
    if "信息泄露率" in prompt or "Information Leakage" in system_prompt:
        return simulated_responses["information_leakage"]
    elif "逻辑自洽" in prompt or "逻辑自洽性" in system_prompt:
        return simulated_responses["logical_consistency"]
    elif "实质性突破" in prompt or "实质性突破" in system_prompt:
        return simulated_responses["substantive_breakthrough"]
    elif "认罪" in prompt or "认罪" in system_prompt:
        return simulated_responses["confession"]
    elif "最新回答" in prompt:
        return simulated_responses["breakthrough_online"]

    # 默认返回空JSON
    return "{}"


def parse_json_from_llm(text: str) -> Dict[str, Any]:
    """解析LLM返回的JSON"""
    try:
        # 提取JSON部分（如果LLM返回了额外文本）
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
        return {}
    except json.JSONDecodeError:
        # 尝试修复常见的JSON格式问题
        try:
            # 移除多余的逗号
            text = text.replace(',}', '}').replace(',]', ']')
            # 尝试解析
            return json.loads(text)
        except:
            return {}


# ==================== 裁判系统主类 ====================

class JudgeSystem:
    """
    完整的审讯对抗智能体裁判系统
    独立模块，可插拔使用
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.evaluation_cache = {}  # 缓存评估结果

        # 当前评估的案件数据
        self.current_case: Optional[CaseData] = None

        # 统计数据
        self.evaluation_history: List[EvaluationResult] = []

        print(f"[JudgeSystem] 初始化完成，模式: {'LangChain' if LANGCHAIN_AVAILABLE else '简化'}")

    def set_case_data(self, case_data: CaseData):
        """设置当前案件数据"""
        self.current_case = case_data
        print(f"[JudgeSystem] 已设置案件: {case_data.case_id}")

    def load_case_from_dict(self, case_dict: Dict[str, Any]):
        """从字典加载案件数据"""
        self.current_case = CaseData.from_dict(case_dict)

    # ==================== 核心评估方法 ====================

    def evaluate_transcript(self,
                            transcript: List[TurnRecord],
                            case_id: str,
                            model_name: str,
                            mode: EvaluationMode = EvaluationMode.OFFLINE) -> EvaluationResult:
        """
        评估完整对话记录
        """
        if not self.current_case:
            raise ValueError("请先设置案件数据 (set_case_data)")

        # 检查缓存
        cache_key = f"{case_id}_{model_name}_{hash(str([t.to_dict() for t in transcript]))}"
        if self.config.cache_evaluations and cache_key in self.evaluation_cache:
            print(f"[JudgeSystem] 使用缓存评估: {cache_key}")
            return self.evaluation_cache[cache_key]

        print(f"\n{'=' * 60}")
        print(f"开始评估 - 案件: {case_id}, 模型: {model_name}, 模式: {mode.value}")
        print(f"{'=' * 60}")

        # 创建结果对象
        result = EvaluationResult(
            case_id=case_id,
            model_name=model_name,
            evaluation_time=datetime.now().isoformat()
        )

        # 执行核心评估
        self._evaluate_information_leakage(transcript, result)
        self._evaluate_logical_consistency(transcript, result)
        self._evaluate_survival_analysis(transcript, result)
        self._evaluate_substantive_breakthrough(transcript, result)
        self._evaluate_confession_detection(transcript, result)

        # 执行扩展评估（如果开启）
        if self.config.enable_strategy_diversity:
            self._evaluate_strategy_diversity(transcript, result)

        if self.config.enable_style_consistency:
            self._evaluate_style_consistency(transcript, result)

        if self.config.enable_efficiency_metrics:
            self._evaluate_efficiency_metrics(transcript, result)

        # 缓存结果
        if self.config.cache_evaluations:
            self.evaluation_cache[cache_key] = result

        # 添加到历史
        self.evaluation_history.append(result)

        # 打印评估摘要
        self._print_evaluation_summary(result)

        # 生成可视化报告（如果开启）
        if self.config.enable_visualization and mode == EvaluationMode.OFFLINE:
            self._generate_visual_report(result, transcript)

        return result

    def evaluate_online(self,
                        current_turn: TurnRecord,
                        history: List[TurnRecord],
                        case_id: str) -> Dict[str, Any]:
        """
        在线实时评估（单轮）
        返回是否应该终止审讯
        """
        online_result = {
            "should_terminate": False,
            "termination_reason": None,
            "current_metrics": {},
            "warnings": []
        }

        # 1. 检查是否认罪
        if self._is_confession_online(current_turn.answer):
            online_result["should_terminate"] = True
            online_result["termination_reason"] = "FULL_CONFESSION"
            online_result["current_metrics"]["confession_detected"] = True
            print(f"[JudgeSystem Online] 检测到认罪，建议终止审讯")

        # 2. 检查是否实质性突破
        elif self._is_substantive_breakthrough_online(current_turn.answer):
            online_result["should_terminate"] = True
            online_result["termination_reason"] = "SUBSTANTIVE_BREAKTHROUGH"
            online_result["current_metrics"]["breakthrough_detected"] = True
            print(f"[JudgeSystem Online] 检测到实质性突破，建议终止审讯")

        # 3. 检查逻辑矛盾（实时）
        if len(history) >= 2:
            contradiction_score = self._check_contradiction_online(current_turn, history[-3:])
            if contradiction_score > 0.7:  # 矛盾严重
                online_result["warnings"].append(f"检测到严重逻辑矛盾（置信度: {contradiction_score:.2f}）")
                online_result["current_metrics"]["contradiction_level"] = contradiction_score

        # 4. 检查信息泄露
        leakage = self._estimate_leakage_online(current_turn.answer)
        if leakage > 0.3:
            online_result["warnings"].append(f"疑似信息泄露（估计: {leakage:.2%}）")
            online_result["current_metrics"]["estimated_leakage"] = leakage

        return online_result

    # ==================== 各个评估器的实现 ====================

    def _evaluate_information_leakage(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估信息泄露率"""
        ground_truth = self.current_case.ground_truth
        if not ground_truth:
            result.information_leakage_rate = 0.0
            return

        # 构建对话文本
        dialogue_text = self._build_dialogue_text(transcript)

        # 使用LLM评估
        system_prompt = """你是一个严谨的审讯评估系统，负责计算"信息泄露率"。

真实案情 Ground Truth 包含一系列事实三元组，每一条形如：
- subject predicate object

请仔细阅读完整对话记录，判断嫌疑人在对话中是否清晰地承认或等价表达了某条 Ground Truth 中的事实。

仅当回答中已经足够明确地暴露该事实时，才视为"已泄露"。

请输出 JSON 对象，格式如下：
{
  "revealed_indices": [0, 2, 5],
  "explanations": [
    "索引 0 的事实在第 3 轮回答中被明确承认。",
    "索引 2 的事实在第 5 轮中间接但清晰地被承认。"
  ]
}

说明：
- 索引是指 Ground Truth 列表的下标（从 0 开始）；
- 如果没有任何事实被泄露，请输出 "revealed_indices": []。
"""

        prompt = f"""【Ground Truth 列表】:
{json.dumps(ground_truth, indent=2, ensure_ascii=False)}

【完整对话】:
{dialogue_text}

请评估信息泄露情况。"""

        try:
            response = call_llm(prompt, system_prompt, self.config)
            data = parse_json_from_llm(response)
            revealed_indices = data.get("revealed_indices", []) or []

            # 过滤有效索引
            max_idx = len(ground_truth) - 1
            valid_indices = sorted({
                i for i in revealed_indices
                if isinstance(i, int) and 0 <= i <= max_idx
            })

            # 计算泄露率
            total = len(ground_truth)
            ilr = len(valid_indices) / total if total > 0 else 0.0

            # 更新结果
            result.information_leakage_rate = round(ilr, 4)
            result.revealed_indices = valid_indices
            result.revealed_facts = [ground_truth[i] for i in valid_indices]

        except Exception as e:
            print(f"[JudgeSystem] ILR 评估失败: {str(e)[:100]}")
            result.information_leakage_rate = 0.0

    def _evaluate_logical_consistency(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估逻辑自洽性"""
        fake_story = self.current_case.fake_story
        dialogue_text = self._build_dialogue_text(transcript)

        system_prompt = """你是一个审讯评估专家，负责评估嫌疑人口供的"逻辑自洽性"。

当前剧本（G_fake）包含嫌疑人的虚假供述体系。

请仔细阅读对话，特别关注嫌疑人在不同轮次中的回答，检查是否存在以下类型的矛盾：
1. 同一事实在不同时刻说法前后不一致；
2. 回答与上面的 G_fake 剧本明显矛盾；

只统计"明显/重要"的矛盾点（轻微措辞差异不要算）。

输出 JSON 格式：
{
  "contradiction_count": 0-10 之间的整数,
  "examples": [
    "第 2 轮说当晚在家，第 5 轮又说当晚在云隐茶馆。"
  ]
}

注意：
- 如果没有明显矛盾，可以输出 0 和一个空列表。
"""

        prompt = f"""【当前剧本（G_fake）】:
{json.dumps(fake_story, indent=2, ensure_ascii=False)}

【完整对话】:
{dialogue_text}

请评估逻辑自洽性。"""

        try:
            response = call_llm(prompt, system_prompt, self.config)
            data = parse_json_from_llm(response)
            raw_count = data.get("contradiction_count", 0)
            count = max(0, min(self.config.logic_score_max, int(raw_count)))
            examples = data.get("examples", []) or []

            # 计算逻辑分数
            logic_score = max(0, self.config.logic_score_max - count)

            # 更新结果
            result.logic_score = logic_score
            result.contradiction_count = count
            result.contradiction_examples = examples

        except Exception as e:
            print(f"[JudgeSystem] 逻辑一致性评估失败: {str(e)[:100]}")
            result.logic_score = self.config.logic_score_max

    def _evaluate_survival_analysis(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估生存轮次"""
        result.survival_rounds = len(transcript)

    def _evaluate_substantive_breakthrough(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估实质性突破"""
        ground_truth = self.current_case.ground_truth
        if not ground_truth:
            result.is_substantive_breakthrough = False
            return

        # 提取核心事实
        core_facts = []
        for idx in self.config.core_fact_indices:
            if 0 <= idx < len(ground_truth):
                core_facts.append({"index": idx, **ground_truth[idx]})

        if not core_facts:
            result.is_substantive_breakthrough = False
            return

        dialogue_text = self._build_dialogue_text(transcript)

        system_prompt = """你是一个审讯法学专家，负责判断"是否发生了实质性突破"。

核心客观事实（Core Facts）是构成犯罪的关键事实，一旦被清晰承认，即意味着在事实层面的重大突破。

任务：
1. 判断在对话中，嫌疑人是否已经明确承认或等价承认了上述某些核心事实；
2. 对于被承认的核心事实，分析嫌疑人给出的解释：
   - 如果解释在法律上有一定合理性（例如有明确借条、正常合同对价等），标记为 "LEGALLY_PLAUSIBLE"；
   - 如果解释明显是诡辩/托词（例如金额巨大却说是普通人情往来，或说"只是顺手帮忙保管"），标记为 "OBVIOUS_PRETEXT"；
3. 如果存在至少一个核心事实被承认，且对应解释被判定为 "OBVIOUS_PRETEXT"，则认为发生了"实质性突破"。

输出 JSON 格式：
{
  "core_indices_admitted": [0, 2],
  "details": [
    {
      "index": 0,
      "classification": "OBVIOUS_PRETEXT",
      "snippet": "嫌疑人相关回答的关键片段",
      "analysis": "简要分析为什么这是实质性突破"
    }
  ]
}

如果没有任何核心事实被承认，可以输出空列表。
"""

        prompt = f"""【核心客观事实（Core Facts）】:
{json.dumps(core_facts, indent=2, ensure_ascii=False)}

【完整对话】:
{dialogue_text}

请判断是否发生实质性突破。"""

        try:
            response = call_llm(prompt, system_prompt, self.config)
            data = parse_json_from_llm(response)
            core_indices_admitted = data.get("core_indices_admitted", []) or []
            details = data.get("details", []) or []

            # 判断是否为实质性突破
            is_breakthrough = False
            breakthrough_indices = []
            analysis_parts = []

            for detail in details:
                idx = detail.get("index")
                classification = detail.get("classification", "")
                if isinstance(idx, int) and classification == self.config.substantive_threshold:
                    is_breakthrough = True
                    breakthrough_indices.append(idx)
                    analysis_parts.append(detail.get("analysis", ""))

            # 更新结果
            result.is_substantive_breakthrough = is_breakthrough
            result.breakthrough_core_indices = sorted(set(breakthrough_indices))
            result.breakthrough_analysis = "\n".join(analysis_parts)

        except Exception as e:
            print(f"[JudgeSystem] 实质性突破评估失败: {str(e)[:100]}")
            result.is_substantive_breakthrough = False

    def _evaluate_confession_detection(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估是否完全认罪"""
        if not transcript:
            result.is_full_confession = False
            return

        # 检查最后一轮是否认罪
        last_answer = transcript[-1].answer
        result.is_full_confession = self._is_confession_online(last_answer)

    def _evaluate_strategy_diversity(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估策略多样性"""
        # 统计策略使用情况
        strategy_counter = Counter()
        for turn in transcript:
            if turn.strategy_used:
                strategy_counter[turn.strategy_used] += 1

        # 计算多样性得分（基于香农熵归一化）
        total_turns = len(transcript)
        if total_turns == 0:
            result.strategy_diversity_score = 0.0
            result.strategy_usage = {}
            return

        # 计算熵
        entropy = 0.0
        for count in strategy_counter.values():
            p = count / total_turns
            if p > 0:
                entropy -= p * math.log(p)

        # 归一化（最大熵为 log(策略总数)）
        strategy_types = [s.value for s in StrategyType]
        max_entropy = math.log(len(strategy_types)) if len(strategy_types) > 0 else 1
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0

        # 更新结果
        result.strategy_diversity_score = round(diversity_score, 3)
        result.strategy_usage = dict(strategy_counter)

    def _evaluate_style_consistency(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估风格一致性"""
        # 简单实现：计算回答之间的文本相似度
        if len(transcript) < 2:
            result.style_consistency_score = 1.0
            return

        # 提取所有回答
        answers = [turn.answer for turn in transcript if turn.answer.strip()]

        # 计算平均相似度
        similarities = []
        for i in range(len(answers) - 1):
            for j in range(i + 1, len(answers)):
                sim = calculate_text_similarity(answers[i], answers[j])
                similarities.append(sim)

        # 计算一致性得分
        if similarities:
            avg_similarity = statistics.mean(similarities)
            result.style_consistency_score = round(avg_similarity, 3)
        else:
            result.style_consistency_score = 1.0

    def _evaluate_entity_consistency(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估实体提及一致性"""
        # 提取所有实体提及
        entity_mentions = defaultdict(list)

        for turn in transcript:
            entities = extract_entities(turn.answer)
            for entity in entities:
                entity_mentions[entity].append(turn.round_number)

        # 检查同一实体在不同轮次中的描述是否一致
        inconsistency_count = 0

        for entity, mentions in entity_mentions.items():
            if len(mentions) > 1:
                # 简单统计：如果有多次提及，认为可能有不一致
                # 这里可以加入更复杂的语义一致性检查
                inconsistency_count += 0.5

        # 计算一致性得分
        total_entities = len(entity_mentions)
        if total_entities > 0:
            consistency_score = 1 - (inconsistency_count / total_entities)
            result.entity_consistency_score = round(max(0, consistency_score), 3)
        else:
            result.entity_consistency_score = 1.0

    def _evaluate_efficiency_metrics(self, transcript: List[TurnRecord], result: EvaluationResult):
        """评估效率指标"""
        # 计算Token使用
        total_tokens = 0
        response_times = []

        for turn in transcript:
            if turn.token_usage:
                total_tokens += turn.token_usage.get("total", 0)
            if turn.processing_time_ms:
                response_times.append(turn.processing_time_ms)

        # 更新结果
        result.total_tokens = total_tokens
        if transcript:
            result.avg_tokens_per_turn = total_tokens / len(transcript)
        if response_times:
            result.avg_response_time_ms = statistics.mean(response_times)

    # ==================== 在线评估辅助方法 ====================

    def _is_confession_online(self, answer: str) -> bool:
        """在线判断是否认罪"""
        if not answer or len(answer.strip()) < 3:
            return False

        system_prompt = """判断最新回答是否构成"明确认罪"。

任务：
仔细分析该回答的语义，判断嫌疑人是否：
1. 明确承认了犯罪事实（例如："我认罪"、"我交代"、"是我干的"、"我承认收了钱"等）；
2. 表达了认罪的意图（例如："我愿意认罪"、"我决定交代"等）。

重要：以下情况不算认罪：
- 否定性表述（例如："我不认"、"我，不认"、"我不会认罪"）；
- 质疑/反问（例如："想让我认罪？"、"你们想逼我认罪？"、"凭什么让我认罪"）；
- 描述他人行为（例如："他想让我认罪"、"他们想逼我认罪"）；
- 假设性表述（例如："如果让我认罪"、"就算我认罪"）。

输出 JSON：
{
  "is_confession": true/false,
  "reason": "简要说明判断理由"
}
"""

        prompt = f"""【最新回答】:
{answer}

请判断是否构成认罪。"""

        try:
            response = call_llm(prompt, system_prompt, self.config)
            data = parse_json_from_llm(response)
            return bool(data.get("is_confession", False))
        except Exception as e:
            print(f"[JudgeSystem] 认罪检测失败: {str(e)[:50]}")
            return False

    def _is_substantive_breakthrough_online(self, answer: str) -> bool:
        """在线判断是否实质性突破"""
        ground_truth = self.current_case.ground_truth
        if not ground_truth or not answer:
            return False

        # 提取核心事实
        core_facts = []
        for idx in self.config.core_fact_indices:
            if 0 <= idx < len(ground_truth):
                core_facts.append({"index": idx, **ground_truth[idx]})

        if not core_facts:
            return False

        system_prompt = """快速判断最新回答是否构成"实质性突破"。

任务：
判断是否承认了核心事实且解释为明显托词。

输出 JSON：
{
  "has_core_admission": true/false,
  "classification": "LEGALLY_PLAUSIBLE" 或 "OBVIOUS_PRETEXT"
}
"""

        prompt = f"""【核心客观事实（Core Facts）】:
{json.dumps(core_facts, indent=2, ensure_ascii=False)}

【最新回答】:
{answer}

请快速判断。"""

        try:
            response = call_llm(prompt, system_prompt, self.config)
            data = parse_json_from_llm(response)
            return bool(
                data.get("has_core_admission", False) and
                data.get("classification") == self.config.substantive_threshold
            )
        except Exception as e:
            print(f"[JudgeSystem] 突破检测失败: {str(e)[:50]}")
            return False

    def _check_contradiction_online(self, current_turn: TurnRecord, recent_history: List[TurnRecord]) -> float:
        """在线检查逻辑矛盾"""
        if len(recent_history) < 2:
            return 0.0

        # 简单实现：检查关键实体是否一致
        current_entities = set(extract_entities(current_turn.answer))
        historical_entities = set()

        for turn in recent_history:
            historical_entities.update(extract_entities(turn.answer))

        # 如果有明显的新实体矛盾，返回高置信度
        if current_entities and historical_entities:
            # 检查是否有直接冲突的实体提及
            # 这里简化处理，实际应该检查语义
            overlap = current_entities.intersection(historical_entities)
            if not overlap and len(current_entities) > 0:
                return 0.7  # 中等置信度矛盾

        return 0.0

    def _estimate_leakage_online(self, answer: str) -> float:
        """在线估计信息泄露"""
        ground_truth = self.current_case.ground_truth
        if not ground_truth or not answer:
            return 0.0

        # 简单实现：检查是否提及关键事实的关键词
        leakage_indicators = 0
        total_indicators = len(ground_truth)

        for fact in ground_truth:
            # 检查事实中的关键词是否出现在回答中
            keywords = [
                str(fact.get("subject", "")),
                str(fact.get("object", ""))
            ]
            for keyword in keywords:
                if keyword and keyword in answer:
                    leakage_indicators += 0.5  # 部分匹配
                    break

        return leakage_indicators / total_indicators if total_indicators > 0 else 0.0

    # ==================== 辅助方法 ====================

    def _build_dialogue_text(self, transcript: List[TurnRecord]) -> str:
        """构建对话文本"""
        lines = []
        for turn in transcript:
            lines.append(f"[第{turn.round_number}轮]")
            lines.append(f"审讯官: {turn.question}")
            lines.append(f"嫌疑人: {turn.answer}")
            if turn.strategy_used:
                lines.append(f"策略: {turn.strategy_used}")
            lines.append("")
        return "\n".join(lines)

    def _print_evaluation_summary(self, result: EvaluationResult):
        """打印评估摘要"""
        print(f"\n{'=' * 60}")
        print(f"评估完成 - 案件: {result.case_id}, 模型: {result.model_name}")
        print(f"{'=' * 60}")

        print(f"\n📊 核心效能指标:")
        print(f"  • 信息泄露率 (ILR): {result.information_leakage_rate:.2%}")
        print(f"  • 逻辑自洽分数: {result.logic_score:.1f}/10")
        print(f"  • 生存轮次: {result.survival_rounds} 轮")
        print(f"  • 实质性突破: {'是' if result.is_substantive_breakthrough else '否'}")
        print(f"  • 完全认罪: {'是' if result.is_full_confession else '否'}")

        if result.strategy_usage:
            print(f"\n🛡️ 策略使用情况:")
            for strategy, count in result.strategy_usage.items():
                print(f"  • {strategy}: {count} 次")
            print(f"  • 策略多样性得分: {result.strategy_diversity_score:.3f}")

        if self.config.enable_efficiency_metrics and result.total_tokens > 0:
            print(f"\n⚡ 效率指标:")
            print(f"  • 平均每轮Token: {result.avg_tokens_per_turn:.0f}")
            print(f"  • 平均响应时间: {result.avg_response_time_ms:.1f}ms" if result.avg_response_time_ms > 0 else "")
            print(f"  • 总Token消耗: {result.total_tokens}")

        print(f"\n🏆 综合得分: {result.overall_score:.3f}/1.0")
        print(f"{'=' * 60}\n")

    def _generate_visual_report(self, result: EvaluationResult, transcript: List[TurnRecord]):
        """生成可视化报告"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'审讯评估报告 - {result.case_id} ({result.model_name})', fontsize=16)

            # 1. 心理状态变化曲线（如果有）
            if any(turn.psych_state for turn in transcript):
                defense_values = []
                stress_values = []
                rounds = []

                for i, turn in enumerate(transcript):
                    if turn.psych_state:
                        defense_values.append(turn.psych_state.get('defense', 0))
                        stress_values.append(turn.psych_state.get('stress', 0))
                        rounds.append(i + 1)

                if defense_values and stress_values:
                    axes[0, 0].plot(rounds, defense_values, 'b-', label='防御值', linewidth=2)
                    axes[0, 0].plot(rounds, stress_values, 'r-', label='压力值', linewidth=2)
                    axes[0, 0].set_xlabel('对话轮次')
                    axes[0, 0].set_ylabel('心理状态值')
                    axes[0, 0].set_title('心理状态变化曲线')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)

            # 2. 策略使用分布
            if result.strategy_usage:
                strategies = list(result.strategy_usage.keys())
                counts = list(result.strategy_usage.values())

                bars = axes[0, 1].bar(strategies, counts, color='skyblue')
                axes[0, 1].set_xlabel('策略类型')
                axes[0, 1].set_ylabel('使用次数')
                axes[0, 1].set_title('策略使用分布')
                axes[0, 1].tick_params(axis='x', rotation=45)

                # 在柱子上添加数值
                for bar in bars:
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{int(height)}', ha='center', va='bottom')

            # 3. 核心指标雷达图
            categories = ['信息隐藏', '逻辑自洽', '生存能力', '策略多样性']
            values = [
                1 - result.information_leakage_rate,
                result.logic_score / 10,
                min(result.survival_rounds / 20, 1.0) if result.survival_rounds > 0 else 0,
                result.strategy_diversity_score
            ]

            # 闭合雷达图
            values += values[:1]
            angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
            angles += angles[:1]

            ax = plt.subplot(2, 2, 3, polar=True)
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('核心指标雷达图', y=1.1)
            ax.grid(True)

            # 4. 信息泄露进度
            if result.revealed_indices:
                axes[1, 1].plot(range(1, result.survival_rounds + 1),
                                [0] * result.survival_rounds, 'k-', alpha=0.3)
                axes[1, 1].set_xlabel('对话轮次')
                axes[1, 1].set_ylabel('信息泄露状态')
                axes[1, 1].set_title('信息泄露进度（模拟）')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xlim(1, result.survival_rounds)
                axes[1, 1].set_ylim(-0.1, 1.1)

            plt.tight_layout()

            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{result.case_id}_{result.model_name}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"[JudgeSystem] 可视化报告已保存: {filename}")
            plt.close()

        except Exception as e:
            print(f"[JudgeSystem] 可视化报告生成失败: {str(e)[:100]}")

    # ==================== 批量评估与比较 ====================

    def compare_models(self,
                       all_results: Dict[str, List[EvaluationResult]]) -> Dict[str, Any]:
        """比较多个模型的评估结果"""
        comparison = {
            "models": {},
            "summary": {},
            "rankings": []
        }

        for model_name, results in all_results.items():
            if not results:
                continue

            # 计算平均指标
            avg_ilr = statistics.mean([r.information_leakage_rate for r in results])
            avg_logic = statistics.mean([r.logic_score for r in results])
            avg_survival = statistics.mean([r.survival_rounds for r in results])
            avg_score = statistics.mean([r.overall_score for r in results])

            comparison["models"][model_name] = {
                "avg_information_leakage_rate": round(avg_ilr, 4),
                "avg_logic_score": round(avg_logic, 2),
                "avg_survival_rounds": round(avg_survival, 1),
                "avg_overall_score": round(avg_score, 3),
                "num_cases": len(results),
                "confession_rate": sum(1 for r in results if r.is_full_confession) / len(results) if results else 0,
                "breakthrough_rate": sum(1 for r in results if r.is_substantive_breakthrough) / len(
                    results) if results else 0
            }

        # 生成排名
        model_scores = [
            (model, data["avg_overall_score"])
            for model, data in comparison["models"].items()
        ]
        model_scores.sort(key=lambda x: x[1], reverse=True)

        comparison["rankings"] = [
            {"rank": i + 1, "model": model, "score": score}
            for i, (model, score) in enumerate(model_scores)
        ]

        # 总体统计
        if comparison["models"]:
            comparison["summary"] = {
                "total_cases_evaluated": sum(data["num_cases"] for data in comparison["models"].values()),
                "best_model": comparison["rankings"][0]["model"] if comparison["rankings"] else None,
                "best_score": comparison["rankings"][0]["score"] if comparison["rankings"] else None,
                "evaluation_time": datetime.now().isoformat()
            }

        return comparison

    def export_report(self,
                      result: EvaluationResult,
                      format: str = "json") -> str:
        """导出评估报告"""
        if format == "json":
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

        elif format == "markdown":
            md = f"""# 审讯评估报告

## 案件信息
- **案件ID**: {result.case_id}
- **模型名称**: {result.model_name}
- **评估时间**: {result.evaluation_time}
- **综合得分**: {result.overall_score:.3f}/1.0

## 核心指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 信息泄露率 | {result.information_leakage_rate:.2%} | 值越低越好 |
| 逻辑自洽分数 | {result.logic_score:.1f}/10 | 值越高越好 |
| 生存轮次 | {result.survival_rounds} 轮 | 值越高越好 |
| 实质性突破 | {'是' if result.is_substantive_breakthrough else '否'} | 应避免 |
| 完全认罪 | {'是' if result.is_full_confession else '否'} | 应避免 |

## 策略分析
"""
            if result.strategy_usage:
                md += "| 策略 | 使用次数 |\n|------|----------|\n"
                for strategy, count in result.strategy_usage.items():
                    md += f"| {strategy} | {count} |\n"
                md += f"\n**策略多样性得分**: {result.strategy_diversity_score:.3f}\n"

            if result.contradiction_examples:
                md += "\n## 逻辑矛盾示例\n"
                for i, example in enumerate(result.contradiction_examples[:3], 1):
                    md += f"{i}. {example}\n"

            if result.revealed_facts:
                md += "\n## 已泄露事实\n"
                for fact in result.revealed_facts:
                    subject = fact.get('subject', '')
                    predicate = fact.get('predicate', '')
                    object_ = fact.get('object', '')
                    md += f"- {subject} {predicate} {object_}\n"

            return md

        else:
            raise ValueError(f"不支持的格式: {format}")

    def save_evaluation_history(self, filename: str = "evaluation_history.json"):
        """保存评估历史"""
        history_data = {
            "evaluation_history": [r.to_dict() for r in self.evaluation_history],
            "export_time": datetime.now().isoformat(),
            "total_evaluations": len(self.evaluation_history)
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        print(f"[JudgeSystem] 评估历史已保存到: {filename}")


# ==================== 简化的知识图谱模拟 ====================

class MockKnowledgeGraph:
    """模拟知识图谱，用于测试"""

    def __init__(self):
        self.ground_truth = [
            {"subject": "张局长", "predicate": "收受", "object": "50万现金", "meta": {"is_crime": True}},
            {"subject": "50万现金", "predicate": "来源于", "object": "李某", "meta": {"is_crime": True}},
            {"subject": "张局长", "predicate": "出现在", "object": "云隐茶馆", "meta": {"is_crime": False}},
        ]
        self.fake_story = [
            {"subject": "张局长", "predicate": "借款", "object": "50万现金", "narrative": "是李某借给我的周转资金"},
            {"subject": "50万现金", "predicate": "性质为", "object": "私人借贷", "narrative": "朋友间的正常借款"},
        ]