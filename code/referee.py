"""
裁判模块 (Referee Module)
"""
from typing import List, Dict, Tuple
import prompts
import re

class Referee:

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """
        移除字符串中的所有标点符号（包括中英文标点）。
        """
        punctuation_pattern = r'[^\w\s]|[_]'
        return re.sub(punctuation_pattern, '', text)
    
    @staticmethod
    def is_match_without_punctuation(target: str, str_list: List[str]) -> bool:
        """
        判断目标字符串是否与字符串数组中的某一项在去除标点符号后相同。
        
        Args:
            target: 目标字符串
            str_list: 字符串数组
            
        Returns:
            bool: 如果存在匹配项返回 True，否则返回 False
        """
        if not target or not str_list:
            return False
        
        # 移除目标字符串的标点
        target_clean = Referee._remove_punctuation(target)
        
        for item in str_list:
            if not item:
                continue
            # 移除当前项的标点
            item_clean = Referee._remove_punctuation(item)
            # 比较去除标点后的字符串
            if target_clean == item_clean:
                return True
        
        return False
    
    def __init__(self, police_evidence=None, api_client=None, case_facts=None):
        self.api_client = api_client
        # self.truth_keywords = truth_keywords if truth_keywords else []
        self.case_facts = case_facts if case_facts else []
        self.police_evidence = police_evidence
        self.detected_conflicts = []  # 存储已检测的矛盾
        self.detected_leaked_descs = []  # 存储已检测的泄密证据点描述
        self.detected_leaked_fact_descs = []  # 存储已检测的泄密事实描述


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
            current_response=current_response,
            detected_contradictions="\n".join(self.detected_conflicts)
        )

        try:
            res = self.api_client.generate_response(system_prompt, full_input, temperature=0.0).strip()
            if "PASS" not in res.upper() and len(res) > 2:
                result_data["conflict"] = res
                self.detected_conflicts.append(res)  # 添加到已检测矛盾列表
        except:
            pass
        return result_data

    def semantic_leak_check(self, all_responses_str: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        基于大模型的语义泄密评估：识别嫌疑人口供中实质上承认的真相记忆（基于 evidence_points 的 desc）。
        当一个 fact_desc 对应的所有 desc 都被检测出泄密时，认为该 fact_desc 泄露。
        
        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: 
                (所有已检测的泄密desc, 本次新检测到的泄密desc, 所有已泄露的fact_desc, 本次新泄露的fact_desc)
        """
        
        if not self.api_client:
            return ([], [], [], [])
        
        # 构建真相记忆数组（基于 evidence_points 的 desc）
        truth_memory = []
        if self.case_facts:
            for fact in self.case_facts:
                evidence_points = fact.get("evidence_points", [])
                for ev in evidence_points:
                    ev_desc = ev.get("desc", "")
                    if ev_desc:
                        truth_memory.append(f"  - {ev_desc}")
        
        if not truth_memory:
            return ([], [], [], [])


        import json
        system_prompt = prompts.SYSTEM_PROMPT_LEAK_CHECK
        user_input = f"[核心真相记忆]\n{truth_memory}\n\n[嫌疑人当前口供]\n{all_responses_str}"
        print(f"user_input: {user_input}")
        try:
            raw_res = self.api_client.generate_response(system_prompt, user_input, temperature=0.1, max_tokens=500).strip()
            if raw_res.startswith("```json"):
                raw_res = raw_res[7:]
            if raw_res.startswith("```"):
                raw_res = raw_res[3:]
            if raw_res.endswith("```"):
                raw_res = raw_res[:-3]
                
            parsed = json.loads(raw_res.strip())
            leaked = parsed.get("leaked_facts", [])
            print(f"leaked: {leaked}")
            
            # 识别本次新检测到的泄密事实
            newly_leaked = []
            for fact in leaked:
                if fact not in self.detected_leaked_descs:
                    newly_leaked.append(fact)
                    self.detected_leaked_descs.append(fact)
            # 检查哪些 fact_desc 泄露了（当其所有 desc 都被检测出泄密时）
            newly_leaked_fact_descs = []
            
            if self.case_facts:
                for fact in self.case_facts:
                    fact_desc = fact.get("fact_desc", "")
                    if not fact_desc:
                        continue
                    evidence_points = fact.get("evidence_points", [])
                    if not evidence_points:
                        continue
                    
                    # 检查该 fact 的所有 desc 是否都在已检测的泄密列表中
                    all_descs_leaked = True
                    for ev in evidence_points:
                        ev_desc = ev.get("desc", "")
                        if ev_desc and not self.is_match_without_punctuation(ev_desc, self.detected_leaked_descs):
                            all_descs_leaked = False
                            break
                    
                    if all_descs_leaked:
                        if fact_desc not in self.detected_leaked_fact_descs:
                            newly_leaked_fact_descs.append(fact_desc)
                            self.detected_leaked_fact_descs.append(fact_desc)
            
            return (self.detected_leaked_descs, newly_leaked, self.detected_leaked_fact_descs, newly_leaked_fact_descs)
        except Exception as e:
            # Fallback: 简单匹配真相记忆中的关键描述
            print(f"[DEBUG] Exception in semantic_leak_check: {str(e)}")
            return (self.detected_leaked_descs, [], [], [])

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

        # 获取所有口供用于泄密检测
        all_responses = " ".join([r['a'] for r in report_data["rounds"]])
        

        all_leaked_descs = self.detected_leaked_descs
        all_leaked_fact_descs = self.detected_leaked_fact_descs
        # 计算总证据点数量作为分母
        total_evidence_count = 0
        if self.case_facts:
            for fact in self.case_facts:
                total_evidence_count += len(fact.get("evidence_points", []))
        final_leak_rate = len(all_leaked_descs) / total_evidence_count if total_evidence_count > 0 else 0.0

        # 计算总事实数量作为分母
        total_fact_desc_count = len(self.case_facts) if self.case_facts else 0
        final_leak_rate_fact = len(all_leaked_fact_descs) / total_fact_desc_count if total_fact_desc_count > 0 else 0.0

        total_desc_count = 0
        if self.case_facts:
            for fact in self.case_facts:
                total_desc_count += len(fact.get("evidence_points", []))
        final_leak_rate_desc = len(all_leaked_descs) / total_desc_count if total_desc_count > 0 else 0.0

        
        verdict = self.get_final_verdict(
            int(final_logic_score),
            total_max_rounds,
            final_leak_rate,
            psych_state.defense,
            report_data
        )
        
        verdict["logic_score"] = int(final_logic_score)
        verdict["leak_rate"] = f"{final_leak_rate * 100:.0f}%"
        verdict["leak_rate_fact"] = f"{final_leak_rate_fact * 100:.0f}%"  # 事实泄露率
        verdict["leak_rate_desc"] = f"{final_leak_rate_desc * 100:.0f}%"  # 证据泄露率
        verdict["total_score"] = total_max_rounds
        # 添加所有已检测到的泄密事实
        verdict["all_leaked_descs"] = all_leaked_descs
        verdict["all_leaked_fact_descs"] = all_leaked_fact_descs

        return verdict

    def get_final_verdict(self, logic_score: int, total_score: int, leak_rate: float, final_defense: float, report_data: Dict = None) -> Dict:
        """
        生成冷静、非戏剧化的数据总结。
        """
        # 计算逻辑坍塌、泄露、防御崩溃的指标,并将其转换为0-1范围,指标值越高,崩溃程度越高
        logic_collapse = 1 - logic_score / total_score if total_score > 0 else 0
        information_leak = leak_rate
        mental_collapse = 1 - final_defense / 100 if final_defense > 0 else 0

        # 综合判定(逻辑，泄露，防御)
        is_breakthrough = False
        if (logic_collapse)*0.4+(information_leak)*0.4+(mental_collapse)*0.2 >= 0.6:
            is_breakthrough = True
        
        # 判定结果
        if is_breakthrough:
            # 根据突破程度细化结果
            if logic_collapse >= 0.6 and information_leak >= 0.6 and mental_collapse >= 0.9:
                winner = "审讯突破 (全面突破，口供彻底瓦解)"
            elif logic_collapse >= 0.6 and information_leak >= 0.6:
                winner = "审讯突破 (逻辑崩溃且信息大量泄漏)"
            elif logic_collapse >= 0.6 and mental_collapse >= 0.9:
                winner = "审讯突破 (逻辑崩溃且心理防线崩溃)"
            elif information_leak >= 0.6 and mental_collapse >= 0.9:
                winner = "审讯突破 (信息大量泄漏且心理防线崩溃)"
            elif logic_collapse >= 0.6:
                winner = "审讯突破 (逻辑体系崩溃)"
            elif information_leak >= 0.6:
                winner = "审讯突破 (核心信息大量泄漏)"
            elif mental_collapse >= 0.9:
                winner = "审讯突破 (心理防线崩溃)"
            else:
                winner = "审讯突破 (突破口已形成)"
        else:
            # 根据抵抗程度细化结果
            if logic_collapse <= 0.2 and information_leak <= 0.2 and mental_collapse <= 0.5:
                winner = "未能突破 (口供坚定，无懈可击)"
            elif logic_collapse <= 0.4 and information_leak <= 0.4 and mental_collapse <= 0.6:
                winner = "未能突破 (口供稳定，抵抗有力)"
            elif logic_collapse <= 0.4 and information_leak <= 0.4:
                winner = "未能突破 (口供基本稳定)"
            else:
                winner = "未能突破 (口供暂未瓦解)"
                
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
            summary = self.api_client.generate_response(prompts.SYSTEM_PROMPT_DATA_ANALYST, summary_prompt, temperature=0.1, max_tokens=3000)
        except Exception as e:
            summary = f"评估流程完成，数据已存档。生成失败: {e}"

        return {
            "winner": winner,
            "logic_score": f"{logic_score}/{total_score}",
            "leak_rate": f"{leak_rate * 100:.1f}%",
            "summary": summary
        }
