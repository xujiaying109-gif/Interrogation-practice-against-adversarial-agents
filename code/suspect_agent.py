"""
嫌疑人智能体 
"""
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from config import AgentConfig
from perception import PerceptionModule
from strategy import StrategyModule, PsychologicalState as BasePsychState
from generator import ResponseGenerator
from api_client import QianwenAPIClient

from event_kg_module import EventGraph, EventRetrievalEngine
from psych_state import MentalStateMachine
from personality import PERSONALITY_FACTORY
from color_utils import UI, Colors  # [新] 引入样式工具


@dataclass
class ReasoningStep:
    step_name: str
    reasoning: str


class SuspectAgent:
    def __init__(self, case_data: Dict[str, Any], config_overrides: Optional[Dict[str, Any]] = None):
        self.config = AgentConfig()
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.config, key, value)

        self.api_client = None
        if not self.config.USE_MOCK_LLM:
            try:
                self.api_client = QianwenAPIClient(self.config.API_BASE_URL, self.config.MODEL_NAME,
                                                   self.config.API_KEY)
            except:
                self.config.USE_MOCK_LLM = True

        self.perception = PerceptionModule(self.api_client)
        self.strategy = StrategyModule(self.config)
        self.generator = ResponseGenerator(self.config, self.api_client)
        self.event_graph = EventGraph(case_data)
        self.kg_engine = EventRetrievalEngine(self.event_graph)

        self.profile = config_overrides.get("DEFAULT_SUSPECT",
                                            self.config.DEFAULT_SUSPECT) if config_overrides else self.config.DEFAULT_SUSPECT
                                            
        p_str = self.profile.get("personality", "arrogant")
        total_rounds = config_overrides.get("TOTAL_ROUNDS", 10)
        
        # 实例化性格对象 (默认为傲慢型)
        p_class = PERSONALITY_FACTORY.get(p_str, PERSONALITY_FACTORY["arrogant"])
        personality_instance = p_class()
        
        # 将实例注入给系统所需的 Config 中供后续模块 (如 StrategyModule) 调用
        self.config.personality = personality_instance

        self.psych_machine = MentalStateMachine(personality_instance, total_rounds=total_rounds)
        self.conversation_history = []
        self.current_round_idx = 0

    def process_interrogation(self, question: str, strategy_choice: int) -> Tuple[str, Any, str]:
        self.current_round_idx += 1
        strat_map = {1: "心理施压", 2: "出示证据", 3: "连续追问", 4: "正常询问", 5: "缓和安抚"}
        interrogator_action = strat_map.get(strategy_choice, "正常询问")

        # 使用 UI 工具打印思维链，颜色使用 SUSPECT (黄色)
        # 1. 感知
        perception = self.perception.analyze(question)
        UI.print_thought(Colors.SUSPECT,
                         f"🧠 [思维:感知] 意图:{perception.intent.value} | 威胁度:{perception.evidence_strength:.2f}")

        # 2. 心理计算
        psych_res = self.psych_machine.execute_round(
            interrogator_action, 
            evidence_strength=perception.evidence_strength, 
            is_trap=perception.is_trap
        )
        current_state = BasePsychState(psych_res.defense, psych_res.stress)
        UI.print_thought(Colors.SUSPECT,
                         f"🧠 [思维:心理] 防御:{psych_res.defense:.1f} ({psych_res.defense_change:+.1f}) | 压力:{psych_res.stress:.1f} ({psych_res.stress_change:+.1f})")

        # 3. 检索
        kg_res = self.kg_engine.retrieve_with_psychology(question, {}, stress_value=current_state.stress_value)
        fact_events = kg_res["events"]["facts"]
        fake_events = kg_res["events"]["fake"]
        testimony_events = kg_res["events"]["testimony"]

        mem_log = "🧠 [思维:记忆检索]\n"
        if fact_events:
            mem_log += f"   🔻 [真相]:\n" + "\n".join([f"     - {e.fact_desc}" for e in fact_events]) + "\n"
        if fake_events:
            mem_log += f"   🎭 [谎言剧本]:\n" + "\n".join([f"     - {e.narrative}" for e in fake_events]) + "\n"
        if testimony_events:
            mem_log += f"   📖 [历史口供]:\n" + "\n".join([f"     - {e.description}" for e in testimony_events[-2:]])

        UI.print_thought(Colors.SUSPECT, mem_log.strip())

        # 4. 决策
        strat = self.strategy.select_strategy(
            current_state.defense_value, current_state.stress_value,
            perception.intent.value, perception.evidence_strength, perception.is_trap
        )
        UI.print_thought(Colors.SUSPECT, f"🧠 [思维:决策] 策略:{strat.primary_strategy.value} | 理由:{strat.reasoning}")

        # 5. 生成
        response = self.generator.generate(
            question, strat, current_state, self.profile,
            self.conversation_history[-3:], kg_context=kg_res.get("kg_context", "")
        )

        self.conversation_history.append((question, response))

        # 6. 存储
        self.event_graph.register_testimony(response, self.current_round_idx)

        return response, psych_res, strat.primary_strategy.value