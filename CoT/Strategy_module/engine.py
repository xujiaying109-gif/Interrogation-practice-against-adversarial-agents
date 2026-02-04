from typing import Dict, Any, List
import logging
import sys
import os

# Add project root to path if needed for imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from .domain import Strategy, StatusLabel, DeceptionStrategy
# Assuming preception_module is in python path
try:
    from CoT.preception_module.domain import Intent, PerceptionResult, RetrievedKnowledge
except ImportError:
    try:
        from preception_module.domain import Intent, PerceptionResult, RetrievedKnowledge
    except ImportError:
        from preception_module.cot_new.domain import Intent, PerceptionResult, RetrievedKnowledge

logger = logging.getLogger(__name__)

class StrategyEngine:
    """策略选择引擎"""

    def __init__(self):
        # 策略矩阵定义
        self.strategy_matrix = self._initialize_strategy_matrix()

    def select_strategy(
            self,
            state_label: str,
            perception: PerceptionResult,
            retrieved_knowledge: RetrievedKnowledge
    ) -> DeceptionStrategy:
        """策略选择"""
        # 尝试使用LLM进行策略选择
        llm_strategy = self._select_strategy_with_llm(state_label, perception, retrieved_knowledge)
        if llm_strategy:
            return llm_strategy

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

        return strategy

    def _select_strategy_with_llm(
            self,
            state_label: str,
            perception: PerceptionResult,
            retrieved_knowledge: RetrievedKnowledge
    ) -> DeceptionStrategy:
        """使用LLM选择策略"""
        prompt = f"""
        你是一个审讯对抗智能体的策略选择模块。请根据当前情况选择最佳的对抗策略。
        
        # 当前状态
        心理状态: {state_label}
        审讯意图: {perception.intent.value}
        证据强度: {perception.evidence_strength}
        是否陷阱: {perception.is_trap}
        
        # 知识检索概况
        真实事件数: {len(retrieved_knowledge.truth_events)}
        虚假事件数: {len(retrieved_knowledge.fake_events)}
        
        # 可选策略
        - direct_denial (直接否认)
        - feign_ignorance (假装不知)
        - rationalization (合理化解释)
        - red_herring (转移视线)
        - partial_admission (部分承认)
        - information_dilution (信息稀释)
        - deflect (转移焦点)
        - counter_attack (反击)
        - full_confession (全盘招供)
        
        请返回如下JSON格式的结果:
        {{
            "primary_strategy": "strategy_name",
            "secondary_strategy": "strategy_name" (可选，无则null),
            "risk_level": "high" | "medium" | "low",
            "confidence": 0.0 to 1.0,
            "focus_areas": ["area1", "area2"],
            "avoid_topics": ["topic1", "topic2"],
            "verbal_cues": ["cue1", "cue2"]
        }}
        """
        
        try:
            response_text = self._call_real_llm(prompt)
            import json
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            data = json.loads(clean_text.strip())
            
            # Convert string to Strategy enum
            primary_str = data.get("primary_strategy", "direct_denial")
            # Handle potential mismatch if LLM returns uppercase or different format
            primary_str = primary_str.lower()
            
            try:
                primary_strategy = Strategy(primary_str)
            except ValueError:
                # Fallback mapping or default
                primary_strategy = Strategy.DIRECT_DENIAL
                
            secondary_strategy = None
            if data.get("secondary_strategy"):
                try:
                    secondary_strategy = Strategy(data.get("secondary_strategy").lower())
                except ValueError:
                    pass
            
            return DeceptionStrategy(
                primary_strategy=primary_strategy,
                secondary_strategy=secondary_strategy,
                focus_areas=data.get("focus_areas", []),
                avoid_topics=data.get("avoid_topics", []),
                verbal_cues=data.get("verbal_cues", []),
                risk_level=data.get("risk_level", "medium"),
                confidence=float(data.get("confidence", 0.5))
            )
            
        except Exception as e:
            logger.error(f"LLM strategy selection failed: {e}")
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
