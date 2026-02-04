from typing import Dict, Any, List, Tuple
import time
import logging

# Assuming modules are in python path
try:
    from CoT.preception_module.domain import EventGraph, RetrievedKnowledge
    from CoT.Strategy_module.domain import DeceptionStrategy, PsychologicalState, Strategy, StatusLabel
except ImportError:
    # Fallback for relative imports or different path structure
    try:
        from preception_module.domain import EventGraph, RetrievedKnowledge
        from Strategy_module.domain import DeceptionStrategy, PsychologicalState, Strategy, StatusLabel
    except ImportError:
         # Try handling the original (incorrect?) import just in case, or maybe path hacking
        from preception_module.cot_new.domain import EventGraph, RetrievedKnowledge
        from Strategy_module.cot_new.domain import DeceptionStrategy, PsychologicalState, Strategy, StatusLabel

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """回复生成器 - 核心生成逻辑"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.token_usage = []
        self.response_times = []

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
        # 默认使用真实LLM，除非配置明确指定使用Mock
        if self.config.get("use_mock_llm", False):
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
        import sys
        import os
        import importlib.util
        from langchain_core.messages import SystemMessage, HumanMessage

        try:
            # 动态导入 get_llm
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # ../../Interrogation-practice-against-adversarial-agents-main
            project_root = os.path.abspath(os.path.join(current_dir, "../../Interrogation-practice-against-adversarial-agents-main"))
            config_path = os.path.join(project_root, "0.   配置-config（框架）.py")
            
            if not os.path.exists(config_path):
                # Fallback to absolute path if relative path fails
                config_path = r"c:\Users\Lenovo\Desktop\transform\agent_interrogation_CoT\Interrogation-practice-against-adversarial-agents-main\0.   配置-config（框架）.py"

            spec = importlib.util.spec_from_file_location("ConfigModule", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_llm = module.get_llm
            
            llm = get_llm()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback to mock if real LLM fails
            return f"Error: LLM call failed - {str(e)}"
