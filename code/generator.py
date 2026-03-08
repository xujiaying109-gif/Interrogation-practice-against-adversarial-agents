"""
生成模块 - 动态博弈
"""
import json
import random
from typing import Dict, Any, List, Tuple
import prompts
from log_utils import get_logger

logger = get_logger(__name__)


class ResponseGenerator:
    def __init__(self, config, api_client=None):
        self.config = config
        self.use_mock = getattr(config, 'USE_MOCK_LLM', True)
        self.api_client = api_client
        self.last_pattern = ""

    def generate(self, question: str, strategy: Any, psych_state: Any,
                 suspect_profile: Dict[str, Any], conversation_history: List[Tuple[str, str]],
                 kg_context: str = "") -> str:

        if self.use_mock or self.api_client is None:
            return "（Mock模式）我不知道。"

        # Extract personality object to get its specific reaction patterns
        personality_key = suspect_profile.get('personality', 'normal')
        from personality import PERSONALITY_FACTORY
        p_class = PERSONALITY_FACTORY.get(personality_key)
        
        current_pattern_text = "防备警惕：对待问题保持高度怀疑。"
        length_instruction = "【字数建议】：你的回复可以控制在 400 个中文字符以内。"
        
        if p_class:
            p_obj = p_class()
            available_patterns_dicts = p_obj.reaction_patterns
            
            # Filter candidates based on current stress and interrogator strategy
            current_stress = psych_state.stress_value
            interrogator_strategy = strategy.primary_strategy.value # 此时传进来的是 Enum 的 value，但根据系统，应当是具体的名字。
            # 这里strategy是从 SuspectAgent 传进来的 DeceptionStrategy 对象
            # 但是 personality 里面写的 trigger 匹配的是主审官行为（如“连续追问”），而不是嫌疑人的对策。
            # 为了准确，我们这里只用 stress 过滤，或者放宽 trigger_strategy 匹配。
            # 由于 generator 目前没直接拿到审讯官的具体选择名，我们就简单根据 stress 过滤：
            valid_patterns = [
                p["pattern"] for p in available_patterns_dicts
                if current_stress <= p.get("max_stress", 100)
            ]
            if not valid_patterns:
                valid_patterns = [p["pattern"] for p in available_patterns_dicts]

            candidates = [p for p in valid_patterns if p != self.last_pattern]
            if not candidates:
                candidates = valid_patterns  # Fallback
                
            current_pattern_text = random.choice(candidates)
            self.last_pattern = current_pattern_text
            
            # 使用性格类自带的字数与语速指令
            length_instruction = p_obj.get_verbosity_rules(current_stress)

        system_prompt = self._build_system_prompt(
            strategy, psych_state, suspect_profile, kg_context, current_pattern_text, length_instruction
        )

        user_content = f"【审讯官提问】{question}"

        if conversation_history:
            recent_hist = conversation_history[-4:]  # 增加近期对话记忆长度到 4 轮
            hist_str = "\n".join([f"警：{q}\n我：{a}" for q, a in recent_hist])
            user_content = f"【近期连贯对话回顾】\n{hist_str}\n\n{user_content}"
            
            # 【自洽审查】：强制模型关注自身之前的谎言
            if len(recent_hist) > 0:
                last_answer = recent_hist[-1][1]
                user_content += f"\n\n【自洽审查（极其重要）】：请确保你接下来的回答必须与你刚刚的这句口供【{last_answer}】在逻辑上不能出现明显矛盾，你需要圆谎！"

        return self.api_client.generate_response(
            system_prompt=system_prompt,
            user_input=user_content,
            temperature=0.95,
            max_tokens=600  # API Token上限放宽，但通过prompt严格控制台词长度
        )

    def _build_system_prompt(self, strategy, psych_state, profile, kg_context, current_pattern,
                             length_instruction) -> str:
        name = profile.get('name', '张某')
        personality_key = profile.get('personality', 'normal')
        stress = psych_state.stress_value
        
        # 提取核心特质和话术
        from personality import PERSONALITY_FACTORY
        import random
        p_class = PERSONALITY_FACTORY.get(personality_key)
        if p_class:
            # 实例化获取属性
            p_obj = p_class()
            core_traits = p_obj.core_traits
            
            # 动态抽样：每轮随机抽取 1-2 句作为参考，防止复读机
            sampled_phrases = random.sample(p_obj.catchphrases, min(2, len(p_obj.catchphrases)))
            catchphrases_str = "、".join([f'"{c}"' for c in sampled_phrases])
        else:
            core_traits = "作为普通嫌疑人，被审查调查时表现出防备心理。"
            catchphrases_str = "“不知道”、“我不清楚”"

        return prompts.SYSTEM_PROMPT_SUSPECT.format(
            name=name,
            personality=personality_key,
            core_traits=core_traits,
            catchphrases=catchphrases_str,
            stress=stress,
            kg_context=kg_context,
            current_pattern=current_pattern,
            length_instruction=length_instruction,
            strategy_str=strategy.primary_strategy.value
        )