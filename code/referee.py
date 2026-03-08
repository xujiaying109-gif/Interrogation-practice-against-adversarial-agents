"""
裁判模块 (Referee Module)
"""
from typing import List, Dict
import prompts

class Referee:
    def __init__(self, police_evidence=None, api_client=None, truth_keywords=None):
        self.api_client = api_client
        self.truth_keywords = truth_keywords if truth_keywords else []
        self.police_evidence = police_evidence

    def verify_realtime_logic(self, current_response: str, history_rounds: List[Dict],
                              max_game_rounds: int = 5,
                              last_logic_score: float = 10.0) -> Dict:
        """
        实时检测机制：
        对比当前回答与历史回答，判断是否存在硬性逻辑矛盾。
        """
        result_data = {"conflict": None}
        if not self.api_client or not history_rounds:
            return result_data

        history_items = []
        for i, r in enumerate(history_rounds):
            ans = r.get('a', "")
            if ans:
                history_items.append(f"第{i + 1}轮口供: {ans}")

        if not history_items:
            return result_data

        history_blob = "\n".join(history_items)

        system_prompt = prompts.SYSTEM_PROMPT_LOGIC_AUDITOR
        full_input = prompts.USER_PROMPT_LOGIC_AUDITOR.format(
            history_blob=history_blob,
            current_response=current_response
        )

        try:
            res = self.api_client.generate_response(system_prompt, full_input, temperature=0.0).strip()
            if "PASS" not in res.upper() and len(res) > 2:
                result_data["conflict"] = res
        except:
            pass
        return result_data

    def semantic_leak_check(self, all_responses_str: str) -> List[str]:
        """
        基于大模型的语义泄密评估：识别嫌疑人口供中实质上承认的清单元素。
        """
        if not self.api_client or not self.truth_keywords:
            return []

        import json
        system_prompt = prompts.SYSTEM_PROMPT_LEAK_CHECK
        user_input = f"[警方保密清单]\n{json.dumps(self.truth_keywords, ensure_ascii=False)}\n\n[嫌疑人全量口供]\n{all_responses_str}"

        try:
            raw_res = self.api_client.generate_response(system_prompt, user_input, temperature=0.1, max_tokens=200).strip()
            
            if raw_res.startswith("```json"):
                raw_res = raw_res[7:]
            if raw_res.startswith("```"):
                raw_res = raw_res[3:]
            if raw_res.endswith("```"):
                raw_res = raw_res[:-3]
                
            parsed = json.loads(raw_res.strip())
            leaked = parsed.get("leaked_keywords", [])
            return leaked if isinstance(leaked, list) else []
        except Exception as e:
            # Fallback to naive matching if LLM parsing fails
            return [kw for kw in self.truth_keywords if kw.lower() in all_responses_str.lower()]

    def adjudicate(self, report_data: Dict, psych_state) -> Dict:
        """
        修正后的结案评估：
        1. 逻辑总分 = 用户设置的最大轮次 (max_game_rounds)
        2. 当前分数 = 总分 - 累计矛盾次数
        """
        # 获取用户设置的总轮次，若获取不到则默认为实际发生的轮次
        total_max_rounds = report_data.get("meta", {}).get("total_rounds", len(report_data["rounds"]))

        # 统计截至目前的矛盾总数
        conflicts_count = sum(1 for r in report_data["rounds"] if r.get("contradiction"))

        # 逻辑分数计算：初始分(总轮次) - 矛盾数
        final_logic_score = total_max_rounds - conflicts_count

        # 关键词泄露计算 (大模型语义判断)
        all_responses = " ".join([r['a'] for r in report_data["rounds"]])
        matched = self.semantic_leak_check(all_responses)
        final_leak_rate = len(matched) / len(self.truth_keywords) if self.truth_keywords else 0.0

        verdict = self.get_final_verdict(
            int(final_logic_score),
            total_max_rounds,
            final_leak_rate,
            psych_state.defense,
            report_data
        )
        
        verdict["logic_score"] = int(final_logic_score)
        verdict["leak_rate"] = f"{final_leak_rate * 100:.0f}%"
        verdict["total_score"] = total_max_rounds

        return verdict

    def get_final_verdict(self, logic_score: int, total_score: int, leak_rate: float, final_defense: float, report_data: Dict = None) -> Dict:
        """
        生成冷静、非戏剧化的数据总结。
        """
        # 胜负判定：逻辑坍塌、泄露过高或防御崩溃
        is_breakthrough = logic_score <= (total_score * 0.6) or leak_rate > 0.6 or final_defense < 10
        winner = "审讯突破 (突破口已形成)" if is_breakthrough else "未能突破 (口供暂未瓦解)"

        # 提取对话实录
        transcript_lines = []
        if report_data and "rounds" in report_data:
            for r in report_data["rounds"]:
                transcript_lines.append(f"第{r.get('round', '?')}轮 | 主审官：{r.get('q','')}")
                transcript_lines.append(f"第{r.get('round', '?')}轮 | 嫌疑人：{r.get('a','')}")
        transcript_str = "\n".join(transcript_lines) if transcript_lines else "（无记录）"
        
        # 提取元数据
        meta = report_data.get("meta", {}) if report_data else {}
        case_title = meta.get("case_title", "未知记录")
        suspect_name = meta.get("suspect_name", "未知人员")
        
        # 提取具体人设名称
        personality_key = meta.get("suspect_personality", "normal")
        from personality import PERSONALITY_FACTORY
        p_class = PERSONALITY_FACTORY.get(personality_key)
        suspect_personality = p_class().name if p_class else personality_key
        
        # 提取案件核心真相
        intel_logs = meta.get("intel_logs", [])
        truth_lines = []
        for i, log in enumerate(intel_logs):
            truth_lines.append(f"【线索 {i+1}】警方视角：{log.get('p_view', '')} -> 实际真相：{log.get('s_view', '')}")
        truth_context = "\n".join(truth_lines) if truth_lines else "（暂无真相记录）"

        summary_prompt = prompts.PROMPT_SUMMARY_EVALUATION.format(
            logic_score=logic_score,
            total_score=total_score,
            leak_rate=leak_rate * 100,
            final_defense=final_defense,
            case_title=case_title,
            suspect_name=suspect_name,
            suspect_personality=suspect_personality,
            truth_context=truth_context,
            transcript=transcript_str
        )
        
        try:
            summary = self.api_client.generate_response(prompts.SYSTEM_PROMPT_DATA_ANALYST, summary_prompt, temperature=0.1)
        except Exception as e:
            summary = f"评估流程完成，数据已存档。生成失败: {e}"

        return {
            "winner": winner,
            "logic_score": f"{logic_score}/{total_score}",
            "leak_rate": f"{leak_rate * 100:.1f}%",
            "summary": summary
        }
