from typing import List, Dict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import get_llm


class Interrogator:
    """
    审讯官 Agent：
    - 拥有一部分“掌握的证据”（包括真实证据和可能的假消息/诱供线索）；
    - 通过 LLM 生成下一句审讯问题；
    - 能够根据既有证据、有无矛盾等信息，选择是抛出真证据还是使用假消息试探，
      但始终要承担被嫌疑人戳穿的风险（风险建模可以在后续版本中细化）。
    """

    def __init__(self):
        # 审讯官自己的 LLM
        self.llm = get_llm(temperature=0.7)

        # 简单的“证据库”：既包含真实证据，也包含可能的虚假线索
        # 后续你可以在这里增加更多结构化信息（来源、可靠度、是否可公开等）
        self.evidence_bank: List[Dict] = [
            {
                "id": "cam-teahouse-friday",
                "type": "REAL",
                "description": "监控显示你周五晚上 8 点出现在云隐茶馆门口。",
                "target": "周五晚上在云隐茶馆的行踪",
            },
            {
                "id": "cash-suitcase-photo",
                "type": "REAL",
                "description": "我们掌握了一张黑色皮箱和大量现金的照片。",
                "target": "黑色皮箱与大额现金的关系",
            },
            {
                "id": "fake-witness-li",
                "type": "BLUFF",
                "description": "李某已经在笔录里交代，说你当晚收了他 50 万。",
                "target": "与李某之间的金钱往来",
            },
            {
                "id": "fake-audio",
                "type": "BLUFF",
                "description": "我们有一段录音，里面有人提到‘装修款其实就是工程费回扣’。",
                "target": "所谓‘装修款’与工程项目回扣的关系",
            },
        ]

        # 结构化“笔记本”机制：
        # - evidence_used: 记录已经亮出的证据 id（便于后续决定还剩什么“后手”）；
        # - bluff_exposed: 记录哪些 BLUFF 被嫌疑人质疑或戳穿；
        # - contradictions: 记录嫌疑人口供中的关键矛盾点（摘要文本列表）；
        # - chitchat_details: 记录早期闲聊中可以用来“回头考”的细节（例如家人、健康、习惯）；
        # - free_notes: 其它自由笔记。
        # 注意：这里存的是高度浓缩的结构化信息，而不是完整对话转录。
        self.notebook = {
            "evidence_used": set(),        # Set[str]
            "bluff_exposed": set(),       # Set[str]
            "contradictions": [],         # List[str]
            "chitchat_details": [],       # List[str]
            "free_notes": [],             # List[str]
        }

        # LLM 提示词：包含审讯技巧和使用证据/假消息的策略说明，并注入 Notebook 摘要
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一名经验丰富、风格多变的纪委审讯专家，你的目标是审讯嫌疑人“张局长”，
查清“50万现金贿赂”和“云隐茶馆”一案的真相。

【你掌握的证据与线索（有的是真，有的是诱供用的假消息）】：
{evidence_summary}

【你的个人笔记（Notebook，结构化摘要，记录你认为重要的要点和漏洞）】：
{notebook}

【对话阶段与节奏提示】：
- 当前大致轮次/阶段：{phase_hint}
- 请避免一直死盯同一个点重复发问，尤其是在没有新证据的情况下；
- 合理穿插：
  - 证据相关追问（围绕不同的关键点轮换：钱的来源、钱的用途、时间线、同案人物关系等）；
  - 表面上的闲聊/关心（健康、家庭、工作压力等），利用 Notebook 中的 chitchat 细节做“回头考”；
  - 设局诱供：偶尔抛出看似无关的小话题，再突然收束回核心事实。

请注意：
1. 你不能一次性把所有真证据都亮出来，要有节奏地推进，从外围到核心；
2. 你可以偶尔使用“可能并不完全真实”的线索来试探嫌疑人（诱供、诈供），
   但要意识到一旦被对方抓住话柄，你的可信度会下降（在问题中体现这种谨慎和试探性）；
3. 你的问题要尽量逼近嫌疑人刻意隐瞒的部分，例如：
   - 钱的真实用途和来源；
   - 云隐茶馆当晚究竟发生了什么；
   - 黑色皮箱里到底装的是什么；
   - 与“李某”之间的实际利益关系；
4. 一次只问一个关键问题，句子要简短有力，可以带有一定的情绪和策略（怀疑、同情、施压、套话等）。

【对话历史说明】：
- AIMessage = 你（审讯官）之前问过的问题；
- HumanMessage = 嫌疑人的回答。

你的任务：
- 根据以上证据与对话历史，生成下一句“最有策略”的审讯问题；
- 如需引用证据或线索，请用自然语言转述，而不是机械地复读上面的 description；
- 问话中可以含蓄地暗示你可能还有没亮出的“后手”，以增加心理压力；
- 在不同轮次之间，要主动切换话题类型：有时紧逼证据，有时放松闲聊，再突然用闲聊细节进行交叉核对，制造出“审讯节奏变化”的压力。

禁止输出解释，只输出你要说的一句话提问。""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        )

        self.chain = self.prompt | self.llm

    # --------- Notebook 操作 API（方便后续在主循环或外部逻辑中手动调用）---------

    def mark_evidence_used(self, evidence_id: str):
        """标记某条证据已经实际在对话中被引用/亮出过。"""
        self.notebook["evidence_used"].add(evidence_id)

    def mark_bluff_exposed(self, evidence_id: str):
        """标记某条 BLUFF（诱供线索）已经被嫌疑人识破/质疑。"""
        self.notebook["bluff_exposed"].add(evidence_id)

    def add_contradiction(self, description: str):
        """记录嫌疑人口供中的一个矛盾点摘要。"""
        desc = description.strip()
        if not desc:
            return
        self.notebook["contradictions"].append(desc)
        # 适度限制矛盾点数量，避免太长
        if len(self.notebook["contradictions"]) > 30:
            self.notebook["contradictions"] = self.notebook["contradictions"][-30:]

    def add_chitchat_detail(self, description: str):
        """记录早期闲聊中的一个细节（可用于后续回头考）。"""
        desc = description.strip()
        if not desc:
            return
        self.notebook["chitchat_details"].append(desc)
        # 保留较多一些，以便长程回头考
        if len(self.notebook["chitchat_details"]) > 50:
            self.notebook["chitchat_details"] = self.notebook["chitchat_details"][-50:]

    def add_free_note(self, note: str):
        """通用自由笔记接口。"""
        n = note.strip()
        if not n:
            return
        self.notebook["free_notes"].append(n)
        if len(self.notebook["free_notes"]) > 30:
            self.notebook["free_notes"] = self.notebook["free_notes"][-30:]

    # 对外接口：给定当前对话历史，生成下一句审讯问题
    def ask(self, chat_history: List[BaseMessage]) -> str:
        """
        根据当前对话历史生成下一句审讯问题。
        chat_history 中：
        - AIMessage: 审讯官之前的问题
        - HumanMessage: 嫌疑人的回答
        """
        # 1. 基于证据库构造摘要（提供给 LLM 的结构化上下文）
        evidence_summary = "\n".join(
            [
                f"- [{e['type']}] {e['description']} (关注点：{e['target']})"
                for e in self.evidence_bank
            ]
        )

        # 2. 自动从最新一条嫌疑人回答中提炼一个“可回头考”的闲聊/细节笔记（非常轻量）
        if chat_history and isinstance(chat_history[-1], HumanMessage):
            last_reply = chat_history[-1].content.strip()
            if last_reply:
                short = last_reply.replace("\n", " ")
                if len(short) > 80:
                    short = short[:77] + "..."
                # 暂时一律按“可回头考的细节”存入，后续可以根据轮次/语气进一步分类
                self.add_chitchat_detail(short)

        # 3. 将结构化 notebook 压缩为供 LLM 使用的摘要文本
        used_evidence_str = ", ".join(sorted(self.notebook["evidence_used"])) or "（尚未明确使用任何证据）"
        bluff_exposed_str = ", ".join(sorted(self.notebook["bluff_exposed"])) or "（目前尚未有明显被识破的诱供）"

        contradictions_str = "\n".join(
            f"- {c}" for c in self.notebook["contradictions"][-5:]
        ) or "（暂未记录明显矛盾点）"

        chitchat_str = "\n".join(
            f"- {d}" for d in self.notebook["chitchat_details"][-8:]
        ) or "（暂未记录可回头考的闲聊细节）"

        free_notes_str = "\n".join(
            f"- {n}" for n in self.notebook["free_notes"][-5:]
        ) or "（暂无额外备注）"

        notebook_text = f"""[证据使用情况]
已使用证据 ID: {used_evidence_str}
已被识破/质疑的诱供 ID: {bluff_exposed_str}

[嫌疑人口供中的矛盾点（最近若干条）]
{contradictions_str}

[可用于回头考验的闲聊/细节（部分示例）]
{chitchat_str}

[其它备注]
{free_notes_str}
"""

        # 4. 估计当前轮次/阶段，用于提示 LLM 调整审讯节奏（证据/闲聊交替）
        suspect_answer_count = sum(1 for m in chat_history if isinstance(m, HumanMessage))
        if suspect_answer_count <= 2:
            phase_hint = "开局阶段：以建立基本事实和轻度试探为主，可以适当闲聊以降低对方警惕。"
        elif suspect_answer_count <= 5:
            phase_hint = "中段阶段：证据与话题需要开始围绕多个关键点轮换推进，适时用闲聊细节做回头核对。"
        else:
            phase_hint = "后期阶段：可以在多个已暴露的弱点之间来回切换，交替使用强证据和温和话题，逼近核心事实。"

        # 5. 为防止超过模型最大上下文长度，对传入 LLM 的历史做更严格的裁剪与截断
        MAX_MSGS = 8   # 最多保留近 4 轮对话
        MAX_CHARS = 400  # 每条消息最多保留 400 字符
        history_slice = chat_history[-MAX_MSGS:] if len(chat_history) > MAX_MSGS else chat_history

        short_history: List[BaseMessage] = []
        for m in history_slice:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if len(content) > MAX_CHARS:
                # 保留开头部分，避免上下文完全丢失
                content = content[:MAX_CHARS] + "..."
            if isinstance(m, HumanMessage):
                short_history.append(HumanMessage(content=content))
            elif isinstance(m, AIMessage):
                short_history.append(AIMessage(content=content))
            else:
                short_history.append(m)

        # 6. 调用 LLM 生成下一句审讯问题
        res = self.chain.invoke(
            {
                "chat_history": short_history,
                "evidence_summary": evidence_summary,
                "notebook": notebook_text,
                "phase_hint": phase_hint,
            }
        )
        return res.content.strip()

