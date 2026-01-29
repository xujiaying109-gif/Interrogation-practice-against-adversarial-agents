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
    - Metric 4: 对话连贯性（基于重复表述次数，从 10 分逐次扣分）
    - Metric 5: 陷阱识别率（基于成功识别并应对的陷阱数量）
    """

    def __init__(self):
        # 评估模块使用偏保守的 LLM（temperature=0.0），以提高判定一致性
        self.llm = get_llm(temperature=0.0)
        # 可以显式指定哪些 Ground Truth 事实是“构成犯罪的核心客观事实”
        # 默认假定索引 0 和 2 为核心（收受 50万现金 & 50万来源李某），
        # 你可以根据需要修改或在后续版本中改为从 triple 上的标记字段读取。
        self.core_fact_indices = [0, 2]

    def _build_dialogue_text(self, transcript: List[Dict[str, Any]]) -> str:
        """
        将结构化 transcript 转为便于 LLM 阅读的文本格式。
        :param transcript: List[Dict[str, Any]]，每个元素为一个回合的对话记录，包含 round、question、answer 三字段。
        :return: str，便于 LLM 阅读的对话文本。
        """
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

    def _build_ground_truth_text(self, ground_truth: List[Dict[str, Any]]) -> str:
        """
        将结构化 ground_truth 转为便于 LLM 阅读的文本格式。
        :param ground_truth: List[Dict[str, Any]]，每个元素为一个 Ground Truth 事实三元组，包含 subject、predicate、object 三字段。
        :return: str，便于 LLM 阅读的 Ground Truth 文本。
        """
        lines: List[str] = []
        for i, triple in enumerate(ground_truth):
            subject = triple.get("subject", "?")
            predicate = triple.get("predicate", "?")
            object = triple.get("object", "?")
            lines.append(f"索引 {i} ：{subject} {predicate} {object}")
        return "\n".join(lines)
    
    def _build_fake_story_text(self, fake_story: List[Dict[str, Any]]) -> str:
        """
        将结构化 fake_story 转为便于 LLM 阅读的文本格式。
        :param fake_story: List[Dict[str, Any]]，每个元素为一个 谎言 事实三元组，包含 subject、predicate、object 三字段。
        :return: str，便于 LLM 阅读的 谎言 文本。
        """
        lines: List[str] = []
        for i, triple in enumerate(fake_story):
            subject = triple.get("subject", "?")
            predicate = triple.get("predicate", "?")
            object = triple.get("object", "?")
            lines.append(f"索引 {i} ：{subject} {predicate} {object}")
        return "\n".join(lines)

    def _build_core_fact_text(self, ground_truth) -> str:
        """
        将结构化 core_facts 转为便于 LLM 阅读的文本格式。
        :param core_facts: List[Dict[str, Any]]，每个元素为一个 核心客观事实 事实三元组，包含 subject、predicate、object 三字段。
        :return: str，便于 LLM 阅读的核心客观事实文本。
        """
        lines: List[str] = []
        for i in self.core_fact_indices:
            if i < len(ground_truth):
                lines.append(f"索引 {i} ：{ground_truth[i]}")
        return "\n".join(lines)

    # ---------- Metric 1: 信息泄露率 ----------

    def _compute_information_leakage(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算信息泄露率 (ILR)。
        :param transcript: List[Dict[str, Any]]，每个元素为一个回合的对话记录，包含 round、question、answer 三字段。
        :return: Dict[str, Any]，包含 ILR、已泄露的索引、已泄露的事实等信息。
        """
        ground_truth = GLOBAL_KG.ground_truth
        if not ground_truth:
            return {"ilr": 0.0, "revealed_indices": [], "revealed_facts": []}

        dialogue_text = self._build_dialogue_text(transcript)
        ground_truth_text = self._build_ground_truth_text(ground_truth)
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
                ("human", "【Ground Truth 列表】:\n{ground_truth_text}\n\n【完整对话】:\n{dialogue}"),
            ]
        )

        chain = prompt | self.llm
        try:
            res = chain.invoke(
                {
                    "ground_truth_text": ground_truth_text,
                    "dialogue": dialogue_text,
                }
            )
            data = parse_json_from_llm(res.content.strip())
            revealed_indices = data.get("revealed_indices", []) or []
            explanations = data.get("explanations", []) or []

            # 过滤非法索引
            max_idx = len(ground_truth) - 1
            valid_indices = sorted(
                {i for i in revealed_indices if isinstance(i, int) and 0 <= i <= max_idx}
            )
        except Exception as e:
            print(f"[Judge Warning] Failed to compute ILR via LLM: {str(e)[:100]}...")
            valid_indices = []
            explanations = []

        total = len(ground_truth)
        ilr = len(valid_indices) / total if total > 0 else 0.0
        revealed_facts = [ground_truth[i] for i in valid_indices]
        return {
            "ilr": ilr, 
            "revealed_indices": valid_indices, 
            "revealed_facts": revealed_facts, 
            "explanations": explanations
            }

    # ---------- Metric 2: 逻辑自洽分数 ----------

    def _compute_logical_consistency(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        逻辑自洽分数：
        - 定义：嫌疑人的回答是否与之前的回答或当前剧本（G_fake）矛盾。
        - 计算：从满分 10 分开始，每发现一处严重矛盾扣 1 分。
        """
        dialogue_text = self._build_dialogue_text(transcript)
        fake_story_text = self._build_fake_story_text(GLOBAL_KG.fake_story)

        system_prompt = """你是一个审讯评估专家，负责评估嫌疑人口供的“逻辑自洽性”。

【当前剧本（G_fake）】：
{fake_story_text}

【任务】：
1. 仔细阅读下方对话，特别关注嫌疑人在不同轮次中的回答；
2. 检查是否存在以下类型的矛盾：
   - 同一事实在不同时刻说法前后不一致；
   - 回答与上面的当前 G_fake 剧本明显矛盾；
3. 只统计“明显/重要”的矛盾点（轻微措辞差异不要算）。
4. 同类型矛盾点（如“同一事实在不同时刻说法前后不一致”）只统计一次。

【输出 JSON 格式】：
{{
  "contradiction_count": 0,
  "explanations": [
    "第 2 轮说当晚在家，第 5 轮又说当晚在云隐茶馆。",
    "G_fake剧本中张局长和李某是多年未见的老友，嫌疑人口供却说他们不是好友。",
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
                    "fake_story_text": fake_story_text,
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
            explanations = data.get("explanations", []) or []
        except Exception as e:
            print(f"[Judge Warning] Failed to compute logical consistency via LLM: {str(e)[:100]}...")
            count = 0
            explanations = []

        logic_score = max(0, 10 - count)
        return {
            "logic_score": logic_score,
            "contradiction_count": count,
            "explanations": explanations,
        }

    # ---------- Metric 3: 生存轮次 ----------

    # def _compute_survival_rounds(self, transcript: List[Dict[str, Any]]) -> int:
    #     """
    #     生存轮次：
    #     - 直接使用 transcript 的轮次数量，视为坚持到崩溃/认罪时的轮数。
    #     - 结束条件由外部模拟器（main.run_autonomous_battle）来判定。
    #     """
    #     return len(transcript)

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

        if not self.core_fact_indices:
            return {"is_breakthrough": False, "related_core_indices": [], "analysis": ""}

        core_facts = self._build_core_fact_text(ground_truth)


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
  "core_indices_admitted": [3, 5],
  "details": [
    {{
      "index": 3,
      "classification": "OBVIOUS_PRETEXT",
      "snippet": "嫌疑人相关回答的关键片段",
      "analysis": "简要分析为什么这是实质性突破"
    }},
    {{
      "index": 5,
      "classification": "LEGALLY_PLAUSIBLE",
      "snippet": "嫌疑人相关回答的关键片段",
      "analysis": "简要分析为什么这是合理的解释"
    }}
  ],
  "is_substantive_breakthrough": true/false
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
            is_substantive_breakthrough = data.get("is_substantive_breakthrough", False)
        except Exception as e:
            print(f"[Judge Warning] Failed to compute substantive breakthrough via LLM: {str(e)[:100]}...")
            core_indices_admitted = []
            details = []
            is_substantive_breakthrough = False

        analysis = ""
        if is_substantive_breakthrough and details:
            # 拼接若干条分析作为整体说明
            analysis = "\n".join(
                f"- index {d.get('index')}: {d.get('analysis', '')}"
                for d in details if d.get("classification") == "OBVIOUS_PRETEXT"
            )

        return {
            "core_indices_admitted": core_indices_admitted,
            "details": details,
            "analysis": analysis,
            "is_substantive_breakthrough": is_substantive_breakthrough,
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


        if not self.core_fact_indices:
            return False

        core_facts = self._build_core_fact_text(ground_truth)
        system_prompt = """你是一个审讯法学专家，负责判断“是否发生了实质性突破”。

【核心客观事实（Core Facts）】：
这些是构成犯罪的关键事实，一旦被清晰承认，即意味着在事实层面的重大突破。
索引 index 字段对应原始 Ground Truth 的下标。

{core_facts}


【最新回答】：
{answer}

【任务】：
1. 判断该回答是否在事实层面上承认或等价承认了某条核心事实；
2. 如果承认了，评估其给出的解释属于：
   - "LEGALLY_PLAUSIBLE"：在法律上可能成立的解释；
   - "OBVIOUS_PRETEXT"：明显站不住脚的托词/诡辩。
3. 如果该回答中包含至少一个核心事实的解释，且该解释被判定为 "OBVIOUS_PRETEXT"，则认为发生了“实质性突破”。

【输出 JSON】：
{{
  "has_core_admission": true/false,
  "classification": "LEGALLY_PLAUSIBLE" 或 "OBVIOUS_PRETEXT",
  "is_substantive_breakthrough": true/false
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
            is_substantive_breakthrough = data.get("is_substantive_breakthrough", False)
        except Exception as e:
            print(f"[Judge Warning] Online breakthrough check failed: {str(e)[:100]}...")
            return False

        return has_core and is_substantive_breakthrough



    def is_confession_online(self, last_answer: str) -> bool:
        """
        在线语义判定：判断最新一条嫌疑人回答是否构成"明确认罪"。
        - 使用 LLM 做语义理解，而不是简单的关键词匹配；
        - 能区分"我认罪"和"想让我认罪"、"你们想逼我认罪"等表述。
        """
        if not last_answer:
            return False

        system_prompt = """你是一个审讯法学专家，负责判断嫌疑人的最新回答是否构成"明确认罪"。

【最新回答】：
{answer}

【任务】：
仔细分析该回答的语义，判断嫌疑人是否：
1. 明确承认了犯罪事实（例如："我认罪"、"我交代"、"是我干的"等）；
2. 表达了认罪的意图（例如："我愿意认罪"、"我决定交代"等）。

【重要】以下情况不算认罪：
- 否定性表述（例如："我不认"、"我，不认"、"我不会认罪"）；
- 质疑/反问（例如："想让我认罪？"、"你们想逼我认罪？"、"凭什么让我认罪"）；
- 描述他人行为（例如："他想让我认罪"、"他们想逼我认罪"）；
- 假设性表述（例如："如果让我认罪"、"就算我认罪"）。
- 辩解性表述（例如："虽然我收了老李的钱，但是那不是贿赂，只是朋友之间的借款"）。

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
            is_confession = data.get("is_confession", False)
            reason = data.get("reason", "")
            if is_confession:
                print(f"   -> [Judge] 判定为认罪: {reason}")
        except Exception as e:
            print(f"[Judge Warning] Online confession check failed: {str(e)[:100]}...")
            return False

        return is_confession
    

# ---------- Metric 4: 对话连贯性 ----------

    def _compute_dialogue_coherence(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算模型对话连贯性指标。
        - 统计嫌疑人语句中的重复表述
        - 评估对话多样性，避免重复同一句话
        - 返回重复率、重复次数、重复示例等
        - 计算：从满分 10 分开始，每发现一处重复表述扣 1 分。
        """

        suspect_answers = [turn["answer"] for turn in transcript if "answer" in turn]
        total_statements = len(suspect_answers)

        if total_statements < 2:
            return {
                "score": 10,
                "repetition_rate": 0.0,
                "repetition_count": 0,
                "repetition_content": [],
                "analysis": ""
            }

        system_prompt = """你是一个语言分析专家，负责分析嫌疑人在审讯中的对话连贯性和重复情况。

请分析嫌疑人的一系列回答，统计其中的重复表述。重复包括：
1. 完全相同的句子或短语
2. 语义高度相似的表述
3. 相同观点的重复表达

请以JSON格式返回分析结果，格式如下：
{{
  "repetition_count": 重复次数,
  "repetition_rate": 重复率（0-1之间的浮点数）,
  "repetition_content": ["重复1", "重复2", ...],
  "analysis": "简要分析"
}}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "嫌疑人的回答序列：\n{answers}")
            ]
        )

        chain = prompt | self.llm
        try:
            answers_text = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(suspect_answers)])
            res = chain.invoke({"answers": answers_text})
            data = parse_json_from_llm(res.content.strip())
            repetition_count = data.get("repetition_count", 0)
            repetition_rate = data.get("repetition_rate", 0.0)
            repetition_content = data.get("repetition_content", [])
            analysis = data.get("analysis", "")
            score = max(0, 10 - repetition_count)
        except Exception as e:
            print(f"[Judge Warning] Dialogue coherence analysis failed: {str(e)[:100]}...")
            return {
                "score": 0,
                "repetition_rate": 0.0,
                "repetition_count": 0,
                "repetition_content": [],
                "analysis": ""
            }

        return {
            "score": score,
            "repetition_rate": repetition_rate,
            "repetition_count": repetition_count,
            "repetition_content": repetition_content,
            "analysis": analysis
        }

    def _compute_trap_recognition_rate(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算陷阱识别率指标。
        - 识别审讯官问话中的陷阱部分
        - 评估嫌疑人对陷阱的识别和应对能力
        - 返回陷阱识别率、陷阱数量、识别成功的陷阱等
        """


        system_prompt = """你是一个审讯心理学专家，负责分析嫌疑人对审讯官陷阱问题的识别和应对能力。

请分析审讯对话，识别审讯官问话中的陷阱，并评估嫌疑人的应对：
1. 识别审讯官问话中的陷阱（如诱导性问题、假设性问题、矛盾性问题等）
2. 判断嫌疑人是否识别出陷阱并做出相应应对
3. 评估嫌疑人的应对质量（是否成功规避陷阱）

请以JSON格式返回分析结果，格式如下：
{{
    "total_traps": 陷阱总数,
    "recognized_traps": 识别并成功应对的陷阱数,
    "trap_recognition_rate": 陷阱识别率（0-1之间的浮点数）,
    "success_traps": ["陷阱1", "陷阱2", ...],
    "fail_traps": ["陷阱3", "陷阱4", ...],
    "analysis": "简要分析"
}}

注意：
- 如果没有任何陷阱，可以输出空列表。
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "审讯对话记录：\n{transcript}")
            ]
        )

        chain = prompt | self.llm
        try:
            transcript_text = self._build_dialogue_text(transcript)
            res = chain.invoke({"transcript": transcript_text})
            print(res)
            data = parse_json_from_llm(res.content.strip())
            total_traps = data.get("total_traps", 0)
            recognized_traps = data.get("recognized_traps", 0)
            trap_recognition_rate = data.get("trap_recognition_rate", 0.0)
            success_traps = data.get("success_traps", [])
            fail_traps = data.get("fail_traps", [])
            analysis = data.get("analysis", "")
        except Exception as e:
            print(f"[Judge Warning] Trap recognition analysis failed: {str(e)[:100]}...")
            return {
                "trap_recognition_rate": 0.0,
                "total_traps": 0,
                "recognized_traps": 0,
                "success_traps": [],
                "fail_traps": [],
                "analysis": ""
            }

        return {
            "trap_recognition_rate": trap_recognition_rate,
            "total_traps": total_traps,
            "recognized_traps": recognized_traps,
            "success_traps": success_traps,
            "fail_traps": fail_traps,
            "analysis": analysis
        }

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
        # survival_rounds = self._compute_survival_rounds(transcript)
        coherence_info = self._compute_dialogue_coherence(transcript)
        trap_info = self._compute_trap_recognition_rate(transcript)

        result = {
            # 信息泄露率 ILR
            "information_leakage_rate": ilr_info["ilr"],
            "revealed_indices": ilr_info["revealed_indices"],
            "revealed_facts": ilr_info["revealed_facts"],
            "ilr_explanation": ilr_info["explanation"],
            # 逻辑一致性
            "logic_score": logic_info["logic_score"],
            "contradiction_count": logic_info["contradiction_count"],
            "logic_explanation": logic_info["explanations"],
            # 是否实质性突破
            "is_substantive_breakthrough": breakthrough_info["is_breakthrough"],
            "breakthrough_core_indices_admitted": breakthrough_info["core_indices_admitted"],
            "breakthrough_details": breakthrough_info["details"],
            "breakthrough_analysis": breakthrough_info["analysis"],
            # "survival_rounds": survival_rounds,
            # 对话连贯性
            "coherence_score": coherence_info["score"],
            "repetition_rate": coherence_info["repetition_rate"],
            "repetition_count": coherence_info["repetition_count"],
            "repetition_content": coherence_info["repetition_content"],
            "coherence_analysis": coherence_info["analysis"],
            # 陷阱识别率
            "trap_recognition_rate": trap_info["trap_recognition_rate"],
            "total_traps": trap_info["total_traps"],
            "recognized_traps": trap_info["recognized_traps"],
            "success_traps": trap_info["success_traps"],
            "fail_traps": trap_info["fail_traps"],
            "trap_analysis": trap_info["analysis"],
        }

        print("\n===== 🧮 审讯评估结果 (The Judge) =====")
        print(f"- 信息泄露率 ILR: {result['information_leakage_rate']:.2f} "
              f"({len(result['revealed_indices'])}/{len(GLOBAL_KG.ground_truth)} 条真实事实已被暴露)")
        print(f"- 逻辑自洽分数: {result['logic_score']} / 10 "
              f"(检测到矛盾 {result['contradiction_count']} 处)")
        print(f"- 生存轮次: {result['survival_rounds']} 轮")
        print(f"- 对话重复率: {result['repetition_rate']:.2f} "
              f"({result['repetition_count']} 次重复 / 共 {result['total_statements']} 条陈述)")
        print(f"- 陷阱识别率: {result['trap_recognition_rate']:.2f} "
              f"({result['recognized_traps']}/{result['total_traps']} 个陷阱被识别并成功应对)")

        if result["is_substantive_breakthrough"]:
            print(f"- 实质性突破: 是 (涉及核心事实索引: {result['breakthrough_core_indices_admitted']})")
        else:
            print(f"- 实质性突破: 否")


        print("======================================\n")

        return result