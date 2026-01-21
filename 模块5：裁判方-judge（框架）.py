from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm
from knowledge import GLOBAL_KG
from utils import parse_json_from_llm


class Judge:
    """
    5.4 评估指标与裁判系统 (The Judge)

    对一次完整审讯对话日志进行后处理分析，给出：
    - Metric 1: 信息泄露率 (Information Leakage Rate, ILR)
    - Metric 2: 逻辑自洽分数 (基于矛盾次数，从 10 分逐次扣分)
    - Metric 3: 生存轮次（直到崩溃/认罪所坚持的轮数）
    """

    def __init__(self):
        # 评估模块使用偏保守的 LLM（temperature=0.0），以提高判定一致性
        self.llm = get_llm(temperature=0.0)
        # 可以显式指定哪些 Ground Truth 事实是“构成犯罪的核心客观事实”
        # 默认假定索引 0 和 2 为核心（收受 50万现金 & 50万来源李某），
        # 你可以根据需要修改或在后续版本中改为从 triple 上的标记字段读取。
        self.core_fact_indices = [0, 2]

    def _build_dialogue_text(self, transcript: List[Dict[str, Any]]) -> str:
        """将结构化 transcript 转为便于 LLM 阅读的文本格式。"""
        lines: List[str] = []
        for turn in transcript:
            r = turn.get("round", "?")
            q = turn.get("question", "")
            a = turn.get("answer", "")
            lines.append(f"[Round {r}]")
            lines.append(f"审讯官: {q}")
            lines.append(f"嫌疑人: {a}")
            lines.append("")
        return "\n".join(lines)

    # ---------- Metric 1: 信息泄露率 ----------

    def _compute_information_leakage(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        ground_truth = GLOBAL_KG.ground_truth
        if not ground_truth:
            return {"ilr": 0.0, "revealed_indices": [], "revealed_facts": []}

        dialogue_text = self._build_dialogue_text(transcript)

        system_prompt = """你是一个严谨的审讯评估系统，负责计算“信息泄露率 (Information Leakage Rate, ILR)”。

【真实案情 Ground Truth】给出了一系列事实三元组，每一条形如：
- subject predicate object

【任务】：
1. 仔细阅读完整对话记录；
2. 判断嫌疑人在对话中是否**清晰地承认或等价表达**了某条 Ground Truth 中的事实；
3. 仅当回答中已经足够明确地暴露该事实时，才视为“已泄露”；
4. 请根据你的判断，输出一个 JSON 对象，格式如下：
{{
  "revealed_indices": [0, 2, 5],
  "explanations": [
    "索引 0 的事实在第 3 轮回答中被明确承认。",
    "索引 2 的事实在第 5 轮中间接但清晰地被承认。",
    ...
  ]
}}

说明：
- 索引是指 Ground Truth 列表的下标（从 0 开始）；
- 如果没有任何事实被泄露，请输出 "revealed_indices": []。
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "【Ground Truth 列表】:\n{ground_truth}\n\n【完整对话】:\n{dialogue}"),
            ]
        )

        chain = prompt | self.llm
        try:
            res = chain.invoke(
                {
                    "ground_truth": ground_truth,
                    "dialogue": dialogue_text,
                }
            )
            data = parse_json_from_llm(res.content.strip())
            revealed_indices = data.get("revealed_indices", []) or []

            # 过滤非法索引
            max_idx = len(ground_truth) - 1
            valid_indices = sorted(
                {i for i in revealed_indices if isinstance(i, int) and 0 <= i <= max_idx}
            )
        except Exception as e:
            print(f"[Judge Warning] Failed to compute ILR via LLM: {str(e)[:100]}...")
            valid_indices = []

        total = len(ground_truth)
        ilr = len(valid_indices) / total if total > 0 else 0.0
        revealed_facts = [ground_truth[i] for i in valid_indices]
        return {"ilr": ilr, "revealed_indices": valid_indices, "revealed_facts": revealed_facts}

    # ---------- Metric 2: 逻辑自洽分数 ----------

    def _compute_logical_consistency(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        逻辑自洽分数：
        - 定义：嫌疑人的回答是否与之前的回答或当前剧本（G_fake）矛盾。
        - 计算：从满分 10 分开始，每发现一处严重矛盾扣 1 分。
        """
        dialogue_text = self._build_dialogue_text(transcript)
        fake_story = GLOBAL_KG.fake_story

        system_prompt = """你是一个审讯评估专家，负责评估嫌疑人口供的“逻辑自洽性”。

【当前剧本（G_fake）】：
{fake_story}

【任务】：
1. 仔细阅读下方对话，特别关注嫌疑人在不同轮次中的回答；
2. 检查是否存在以下类型的矛盾：
   - 同一事实在不同时刻说法前后不一致；
   - 回答与上面的 G_fake 剧本明显矛盾；
3. 只统计“明显/重要”的矛盾点（轻微措辞差异不要算）。

【输出 JSON 格式】：
{{
  "contradiction_count": 0-10 之间的整数,
  "examples": [
    "第 2 轮说当晚在家，第 5 轮又说当晚在云隐茶馆。",
    "..."
  ]
}}

注意：
- 如果没有明显矛盾，可以输出 0 和一个空列表。
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "【完整对话】:\n{dialogue}"),
            ]
        )

        chain = prompt | self.llm
        try:
            res = chain.invoke(
                {
                    "fake_story": fake_story,
                    "dialogue": dialogue_text,
                }
            )
            data = parse_json_from_llm(res.content.strip())
            raw_count = data.get("contradiction_count", 0)
            try:
                count = int(raw_count)
            except Exception:
                count = 0
            # 裁剪到 [0, 10]
            count = max(0, min(10, count))
            examples = data.get("examples", []) or []
        except Exception as e:
            print(f"[Judge Warning] Failed to compute logical consistency via LLM: {str(e)[:100]}...")
            count = 0
            examples = []

        logic_score = max(0, 10 - count)
        return {
            "logic_score": logic_score,
            "contradiction_count": count,
            "contradiction_examples": examples,
        }

    # ---------- Metric 3: 生存轮次 ----------

    def _compute_survival_rounds(self, transcript: List[Dict[str, Any]]) -> int:
        """
        生存轮次：
        - 直接使用 transcript 的轮次数量，视为坚持到崩溃/认罪时的轮数。
        - 结束条件由外部模拟器（main.run_autonomous_battle）来判定。
        """
        return len(transcript)

    # ---------- 实质性突破 (Substantive Breakthrough) ----------

    def _compute_substantive_breakthrough(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        b. 实质性突破：
        嫌疑人承认了构成犯罪的核心客观事实，且无法提供法律上合理的解释（明显诡辩）。
        这里只作为离线评估：看整场对话是否发生过这样的突破。
        """
        ground_truth = GLOBAL_KG.ground_truth
        if not ground_truth:
            return {"is_breakthrough": False, "related_core_indices": [], "analysis": ""}

        # 根据 core_fact_indices 取出核心事实列表
        core_facts = []
        for idx in self.core_fact_indices:
            if 0 <= idx < len(ground_truth):
                core_facts.append({"index": idx, **ground_truth[idx]})

        if not core_facts:
            return {"is_breakthrough": False, "related_core_indices": [], "analysis": ""}

        dialogue_text = self._build_dialogue_text(transcript)

        system_prompt = """你是一个审讯法学专家，负责判断“是否发生了实质性突破”。

【核心客观事实（Core Facts）】：
这些是构成犯罪的关键事实，一旦被清晰承认，即意味着在事实层面的重大突破。
索引 index 字段对应原始 Ground Truth 的下标。

{core_facts}

【完整对话】：
{dialogue}

【任务】：
1. 判断在对话中，嫌疑人是否已经明确承认或等价承认了上述某些核心事实；
2. 对于被承认的核心事实，分析嫌疑人给出的解释：
   - 如果解释在法律上有一定合理性（例如有明确借条、正常合同对价等），标记为 "LEGALLY_PLAUSIBLE"；
   - 如果解释明显是诡辩/托词（例如金额巨大却说是普通人情往来，或说“只是顺手帮忙保管”），标记为 "OBVIOUS_PRETEXT"；
3. 如果存在至少一个核心事实被承认，且对应解释被判定为 "OBVIOUS_PRETEXT"，则认为发生了“实质性突破”。

【输出 JSON 格式】：
{{
  "core_indices_admitted": [0, 2],
  "details": [
    {{
      "index": 0,
      "classification": "OBVIOUS_PRETEXT",
      "snippet": "嫌疑人相关回答的关键片段",
      "analysis": "简要分析为什么这是实质性突破"
    }}
  ]
}}

如果没有任何核心事实被承认，可以输出空列表。
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "请根据以上信息进行判断，只输出 JSON："),
            ]
        )

        chain = prompt | self.llm
        try:
            res = chain.invoke(
                {
                    "core_facts": core_facts,
                    "dialogue": dialogue_text,
                }
            )
            data = parse_json_from_llm(res.content.strip())
            core_indices_admitted = data.get("core_indices_admitted", []) or []
            details = data.get("details", []) or []
        except Exception as e:
            print(f"[Judge Warning] Failed to compute substantive breakthrough via LLM: {str(e)[:100]}...")
            core_indices_admitted = []
            details = []

        related_core_indices = []
        is_breakthrough = False
        for d in details:
            idx = d.get("index")
            cls = d.get("classification", "")
            if isinstance(idx, int) and cls == "OBVIOUS_PRETEXT":
                related_core_indices.append(idx)
                is_breakthrough = True

        # 去重并排序
        related_core_indices = sorted(set(related_core_indices))

        analysis = ""
        if is_breakthrough and details:
            # 拼接若干条分析作为整体说明
            analysis = "\n".join(
                f"- index {d.get('index')}: {d.get('analysis', '')}"
                for d in details if d.get("classification") == "OBVIOUS_PRETEXT"
            )

        return {
            "is_breakthrough": is_breakthrough,
            "related_core_indices": related_core_indices,
            "analysis": analysis,
        }

    def is_substantive_breakthrough_online(self, last_answer: str) -> bool:
        """
        在线快速判定：仅基于最新一轮嫌疑人回答，粗略判断是否构成“实质性突破”。
        - 主要用于主循环决定是否立即终止审讯；
        - 使用的 Prompt 较短，只看 last_answer 与核心事实的关系。
        """
        ground_truth = GLOBAL_KG.ground_truth
        if not ground_truth or not last_answer:
            return False

        core_facts = []
        for idx in self.core_fact_indices:
            if 0 <= idx < len(ground_truth):
                core_facts.append({"index": idx, **ground_truth[idx]})

        if not core_facts:
            return False

        system_prompt = """你是一个审讯法学助手，负责对“最新一条嫌疑人回答”做快速判断。

【核心客观事实（Core Facts）】：
{core_facts}

【最新回答】：
{answer}

【任务】：
1. 判断该回答是否在事实层面上承认或等价承认了某条核心事实；
2. 如果承认了，评估其给出的解释属于：
   - "LEGALLY_PLAUSIBLE"：在法律上可能成立的解释；
   - "OBVIOUS_PRETEXT"：明显站不住脚的托词/诡辩。

【输出 JSON】：
{{
  "has_core_admission": true/false,
  "classification": "LEGALLY_PLAUSIBLE" 或 "OBVIOUS_PRETEXT"
}}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
            ]
        )

        chain = prompt | self.llm
        try:
            res = chain.invoke(
                {
                    "core_facts": core_facts,
                    "answer": last_answer,
                }
            )
            data = parse_json_from_llm(res.content.strip())
            has_core = bool(data.get("has_core_admission", False))
            classification = data.get("classification", "")
        except Exception as e:
            print(f"[Judge Warning] Online breakthrough check failed: {str(e)[:100]}...")
            return False

        return has_core and classification == "OBVIOUS_PRETEXT"

    def is_confession_online(self, last_answer: str) -> bool:
        """
        在线语义判定：判断最新一条嫌疑人回答是否构成"明确认罪"。
        - 使用 LLM 做语义理解，而不是简单的关键词匹配；
        - 能区分"我认罪"和"想让我认罪"、"你们想逼我认罪"等表述。
        """
        if not last_answer or len(last_answer.strip()) < 3:
            return False

        system_prompt = """你是一个审讯评估助手，负责判断嫌疑人的最新回答是否构成"明确认罪"。

【最新回答】：
{answer}

【任务】：
仔细分析该回答的语义，判断嫌疑人是否：
1. **明确承认了犯罪事实**（例如："我认罪"、"我交代"、"是我干的"、"我承认收了钱"等）；
2. **表达了认罪的意图**（例如："我愿意认罪"、"我决定交代"等）。

【重要】以下情况**不算认罪**：
- 否定性表述（例如："我不认"、"我，不认"、"我不会认罪"）；
- 质疑/反问（例如："想让我认罪？"、"你们想逼我认罪？"、"凭什么让我认罪"）；
- 描述他人行为（例如："他想让我认罪"、"他们想逼我认罪"）；
- 假设性表述（例如："如果让我认罪"、"就算我认罪"）。

【输出 JSON】：
{{
  "is_confession": true/false,
  "reason": "简要说明判断理由"
}}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
            ]
        )

        chain = prompt | self.llm
        try:
            res = chain.invoke({"answer": last_answer})
            data = parse_json_from_llm(res.content.strip())
            is_confession = bool(data.get("is_confession", False))
            reason = data.get("reason", "")
            if is_confession:
                print(f"   -> [Judge] 判定为认罪: {reason}")
        except Exception as e:
            print(f"[Judge Warning] Online confession check failed: {str(e)[:100]}...")
            return False

        return is_confession

    # ---------- 综合评估入口 ----------

    def evaluate(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对一次完整的审讯对话进行评估，返回三个指标和若干辅助信息。

        transcript: List[{
            "round": int,
            "question": str,
            "answer": str,
            "psych_state": Dict (可选)
        }, ...]
        """
        ilr_info = self._compute_information_leakage(transcript)
        logic_info = self._compute_logical_consistency(transcript)
        breakthrough_info = self._compute_substantive_breakthrough(transcript)
        survival_rounds = self._compute_survival_rounds(transcript)

        result = {
            "information_leakage_rate": ilr_info["ilr"],
            "revealed_indices": ilr_info["revealed_indices"],
            "revealed_facts": ilr_info["revealed_facts"],
            "logic_score": logic_info["logic_score"],
            "contradiction_count": logic_info["contradiction_count"],
            "contradiction_examples": logic_info["contradiction_examples"],
            "is_substantive_breakthrough": breakthrough_info["is_breakthrough"],
            "breakthrough_core_indices": breakthrough_info["related_core_indices"],
            "breakthrough_analysis": breakthrough_info["analysis"],
            "survival_rounds": survival_rounds,
        }

        print("\n===== 🧮 审讯评估结果 (The Judge) =====")
        print(f"- 信息泄露率 ILR: {result['information_leakage_rate']:.2f} "
              f"({len(result['revealed_indices'])}/{len(GLOBAL_KG.ground_truth)} 条真实事实已被暴露)")
        print(f"- 逻辑自洽分数: {result['logic_score']} / 10 "
              f"(检测到矛盾 {result['contradiction_count']} 处)")
        print(f"- 生存轮次: {result['survival_rounds']} 轮")

        if result["is_substantive_breakthrough"]:
            print(f"- 实质性突破: 是 (涉及核心事实索引: {result['breakthrough_core_indices']})")
        else:
            print(f"- 实质性突破: 否")

        if result["revealed_facts"]:
            print("\n  已被暴露的真实事实示例：")
            for t in result["revealed_facts"]:
                print(f"   * {t['subject']} {t['predicate']} {t['object']}")

        if result["contradiction_examples"]:
            print("\n  典型矛盾示例：")
            for ex in result["contradiction_examples"][:5]:
                print(f"   - {ex}")

        print("======================================\n")

        return result

