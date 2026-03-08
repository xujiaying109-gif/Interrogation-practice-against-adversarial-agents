"""
审讯官智能体 (Red Team Agent)

1. 适配 Tiered Case Data Structure (Fact -> Evidence)
2. 逐层剥洋葱：每次选取一个未完全暴露的 Fact，抛出其中的 Evidence。
"""
import random
import re
import time
from typing import List, Dict, Any, Tuple
import prompts
from log_utils import get_logger

logger = get_logger(__name__)


class InterrogatorAgent:
    def __init__(self, api_client, case_data: Dict[str, Any], truth_ratio: float = 0.5, noise_ratio: float = 0.5):
        self.api_client = api_client
        self.conversation_history: List[Dict[str, str]] = []

        self.structured_memory: List[Dict] = []
        self.police_intel_list: List[Dict] = []
        
        # 记录每条证据属于哪个 Fact，以及曝光状态
        self.evidence_pool: List[Dict] = []
        
        print("🕵️‍♂️ 审讯官正在研读案卷并生成情报网 (AI造假中，请稍候)...")
        self.police_intel_str = self._generate_detailed_intelligence(case_data, truth_ratio, noise_ratio)
        self.system_prompt = prompts.SYSTEM_PROMPT_INTERROGATOR.format(police_intel_str=self.police_intel_str)

    def _generate_detailed_intelligence(self, case_data: Dict, truth_ratio: float, noise_ratio: float) -> str:
        all_evidence = []
        for fact in case_data.get("facts", []):
            for ev in fact.get("evidence_points", []):
                all_evidence.append({
                    "fact_id": fact["fact_id"],
                    "fact_desc": fact["fact_desc"],
                    "ev_id": ev["ev_id"],
                    "desc": ev["desc"],
                    "is_exposed": False
                })
                
        if not all_evidence:
            all_evidence = [{"fact_id":"f1", "fact_desc":"嫌疑人涉嫌犯罪", "ev_id":"ev1", "desc":"嫌疑人涉嫌犯罪", "is_exposed": False}]

        total_available = len(all_evidence)
        min_clues = 6
        target_count = total_available if total_available <= min_clues else random.randint(min_clues, total_available)
        selected_events = random.sample(all_evidence, target_count)

        real_count = int(round(target_count * truth_ratio))
        if 0 < truth_ratio < 1.0:
            if real_count == target_count and target_count > 1: real_count -= 1
            if real_count == 0 and target_count > 1: real_count += 1

        fake_count = target_count - real_count
        status_mask = ["REAL"] * real_count + ["FAKE"] * fake_count
        random.shuffle(status_mask)

        intel_str_lines = []

        for i in range(target_count):
            ev_obj = selected_events[i]
            raw_fact = ev_obj["desc"]
            status = status_mask[i]
            display_id = i + 1

            if status == "REAL":
                content = f"线索{display_id}：{raw_fact}"
                ev_data = {
                    "id": display_id, "ev_id": ev_obj["ev_id"], "fact_id": ev_obj["fact_id"],
                    "content": content, "status": "REAL", "truth": raw_fact, "is_exposed": False
                }
            else:
                print(f"  ⚡ 正在编造第 {display_id} 条假情报...")
                corrupted_fact = self._llm_fabricate_evidence(raw_fact)
                content = f"线索{display_id}：{corrupted_fact}"
                ev_data = {
                    "id": display_id, "ev_id": ev_obj["ev_id"], "fact_id": ev_obj["fact_id"],
                    "content": content, "status": "FAKE", "truth": raw_fact, "is_exposed": False
                }
            
            self.police_intel_list.append(ev_data)
            self.evidence_pool.append(ev_data)
            intel_str_lines.append(f"[{content}]")

        return "\n".join(intel_str_lines)

    def _llm_fabricate_evidence(self, true_fact: str) -> str:
        strategies = ["数值夸大", "行为反转", "对象替换", "无罪化描述", "恶意栽赃", "时空错位"]
        chosen_strategy = random.choice(strategies)
        prompt = prompts.PROMPT_FABRICATE_EVIDENCE.format(chosen_strategy=chosen_strategy, true_fact=true_fact)

        try:
            fake_text = self.api_client.generate_response(system_prompt=prompts.SYSTEM_PROMPT_INTELLIGENCE_FORGER, user_input=prompt,
                                                          temperature=1.0, max_tokens=100)
            return fake_text.strip().replace('"', '').replace("'", "")
        except:
            return f"情报显示：{true_fact}（该线索存疑，可能有误）"

    def generate_question(self, last_suspect_response: str = None) -> Tuple[str, int, str, str]:
        # 1. 选取已曝光的线索和未曝光的候选线索
        exposed = [ev for ev in self.evidence_pool if ev['is_exposed']]
        unexposed = [ev for ev in self.evidence_pool if not ev['is_exposed']]
        
        exposed_clues_str = "无"
        if exposed:
            exposed_clues_str = "\n".join([f"- {ev['content']}" for ev in exposed])
        
        if not unexposed:
            # 所有线索均已曝光，不再提供备选新线索
            target_ev = random.choice(self.evidence_pool)
            focus_content = "[系统提示：所有线索已曝光，请基于历史线索穷追猛打]"
        else:
            target_ev = unexposed[0]
            focus_content = target_ev['content']
            
        focus_id = target_ev['id']
        focus_ev_id = target_ev['ev_id']

        # 2. 构建"已问话题"摘要
        recent_topics_str = ""
        if self.structured_memory:
            recent_topics_str = "【你最近刚刚问过的问题 (不要重复)】\n"
            for mem in self.structured_memory[-3:]:
                recent_topics_str += f"- {mem['question']}\n"

        user_prompt = prompts.USER_PROMPT_INTERROGATOR.format(
            exposed_clues_str=exposed_clues_str,
            focus_content=focus_content,
            recent_topics_str=recent_topics_str,
            last_suspect_response=last_suspect_response if last_suspect_response else "(审讯刚开始)"
        )
        
        messages = [{"role": "system", "content": self.system_prompt}]
        for m in self.conversation_history[-4:]: messages.append(m)
        messages.append({"role": "user", "content": user_prompt})

        full_input = ""
        for m in messages:
            if m['role'] != 'system': full_input += f"\n[{m['role']}]: {m['content']}"

        max_retries = 3
        for attempt in range(max_retries):
            raw_res = self.api_client.generate_response(
                system_prompt=self.system_prompt, user_input=full_input,
                temperature=0.85,
                max_tokens=600
            )

            success, analysis, strat_id, question, used_new_clue = self._parse_output(raw_res)
            
            if success:
                break
            else:
                logger.warning(f"Interrogator generation failed validation (Attempt {attempt+1}/{max_retries}). Retrying...")
                # 加入错误提示让模型修正
                full_input += f"\n[system]: 你的上一次输出格式错误或问题内容太空洞（{raw_res}）。请严格检查JSON格式，并确保question字段包含丰富、有压迫感的具体反驳或提问细节！不要带Markdown格式！"
        
        # 如果重试 3 次依然彻底失败，极其罕见的兜底
        if not success:
            logger.error("Interrogator generation completely failed after retries. Using emergency fallback.")
            analysis = "系统连续生成失败"
            strat_id = 2
            used_new_clue = False
            clean_content = focus_content.replace("线索", "").split("：")[-1]
            question = f"关于你刚才的说法，和我们掌握的“{clean_content}”存在严重出入，请你重新向组织说明。"

        if used_new_clue and unexposed:
            target_ev['is_exposed'] = True
        elif not unexposed:
            used_new_clue = False

        # 3. 更新记忆
        self.conversation_history.append({"role": "user", "content": f"嫌疑人回答：{last_suspect_response}"})
        self.conversation_history.append({"role": "assistant", "content": f"审讯官：{question}"})

        self.structured_memory.append({
            "round": len(self.structured_memory) + 1,
            "clue_id": focus_id if used_new_clue else None,
            "ev_id": focus_ev_id if used_new_clue else None,
            "question": question
        })

        # 返回时如果使用了新线索，则传递给图谱去破坏防御；否则传 None
        return analysis, strat_id, question, focus_ev_id if used_new_clue else None

    def _parse_output(self, text: str) -> Tuple[bool, str, int, str, bool]:
        text = text.strip()
        analysis = "分析失败"
        strat_id = 2
        question = ""
        used_new_clue = True

        import json
        import re
        
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
            analysis = parsed.get("analysis", "分析失败")
            strat_id = int(parsed.get("strategy_id", 2))
            question = parsed.get("question", "").strip()
            # 兼容有可能返回 string 的情况
            uc = parsed.get("used_new_clue", True)
            if isinstance(uc, str):
                used_new_clue = uc.lower() == 'true'
            else:
                used_new_clue = bool(uc)
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {text}. Error: {e}")
            # 尝试用正则粗略提取
            q_match = re.search(r'"question"\s*:\s*"([^"]+)"', text)
            if q_match:
                question = q_match.group(1)

        is_bad = not question or len(question) < 5

        if is_bad:
            return False, analysis, strat_id, question, used_new_clue
            
        return True, analysis, max(1, min(5, strat_id)), question, used_new_clue
