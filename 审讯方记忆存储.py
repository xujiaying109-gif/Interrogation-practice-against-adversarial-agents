# interrogator_memory.py
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import json
import re


class InterrogatorMemory:
    """
    审讯官记忆系统 - 核心记忆存储
    存储审讯过程中的所有关键信息
    """

    def __init__(self, suspect_name: str = "张局长"):
        self.suspect_name = suspect_name

        # ============ 核心记忆存储区 ============

        # 1. 对话历史记忆
        self.conversation_history: List[Dict] = []  # 完整对话记录
        self.asked_questions: Set[str] = set()  # 已问过的问题（标准化去重）
        self.asked_questions_raw: List[str] = []  # 原始问题列表（保持顺序）

        # 2. 事实与证据记忆
        self.confirmed_facts: List[Dict] = []  # 嫌疑人已承认的事实
        self.denied_facts: List[Dict] = []  # 嫌疑人否认的事实
        self.evidence_mentioned: List[Dict] = []  # 提到的证据

        # 3. 矛盾与漏洞记忆
        self.contradictions: List[Dict] = []  # 发现的矛盾点
        self.logical_gaps: List[Dict] = []  # 逻辑漏洞
        self.inconsistencies: List[Dict] = []  # 不一致之处

        # 4. 嫌疑人特征记忆
        self.suspect_profile: Dict[str, List] = {
            "家庭情况": [],
            "健康状况": [],
            "工作习惯": [],
            "个人爱好": [],
            "行为特征": [],
            "语言风格": []
        }

        # 5. 时间线记忆
        self.timeline_facts: Dict[str, List] = {}  # 按时间组织的事实
        self.time_contradictions: List[Dict] = []  # 时间矛盾

        # 6. 关键实体记忆
        self.key_entities: Dict[str, Dict] = {}  # 关键人物、地点、物品
        self.relationships: List[Dict] = []  # 关系网络

        # ============ 审讯策略状态 ============
        self.interrogation_phase = "建立关系"  # 当前审讯阶段
        self.pressure_level = 0  # 施压级别 0-10
        self.evidence_used: List[str] = []  # 已使用的证据ID
        self.bluff_attempts: List[Dict] = []  # 诱供尝试记录
        self.pending_inquiries: List[str] = []  # 待追问的问题

        # ============ 统计信息 ============
        self.stats = {
            "total_rounds": 0,
            "admissions_count": 0,
            "denials_count": 0,
            "contradictions_count": 0,
            "pressure_applied": 0,
            "evidence_revealed": 0
        }

    # ============ 核心记忆操作方法 ============

    def add_conversation(self, question: str, answer: str, metadata: Dict = None) -> None:
        """添加完整对话记录到核心记忆"""
        round_num = len(self.conversation_history) + 1

        # 创建对话记录
        record = {
            "round": round_num,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "phase": self.interrogation_phase,
            "pressure_level": self.pressure_level,
            "metadata": metadata or {},
            "analysis": self._analyze_conversation(question, answer, round_num)
        }

        # 存储到核心记忆
        self.conversation_history.append(record)

        # 处理问题记忆
        self._process_question_memory(question, round_num)

        # 处理回答记忆
        self._process_answer_memory(answer, round_num)

        # 更新统计
        self.stats["total_rounds"] = round_num

        print(f"[记忆系统] 第{round_num}轮对话已存储到记忆")

    def _process_question_memory(self, question: str, round_num: int) -> None:
        """处理问题相关记忆"""
        # 原始问题存储
        self.asked_questions_raw.append(question)

        # 标准化去重存储
        normalized = self._normalize_text(question)
        self.asked_questions.add(normalized)

        # 提取问题中的关键信息
        self._extract_question_info(question, round_num)

    def _process_answer_memory(self, answer: str, round_num: int) -> None:
        """处理回答相关记忆"""
        # 事实提取
        facts = self._extract_facts_from_answer(answer, round_num)
        for fact in facts:
            if fact["type"] == "admission":
                self.confirmed_facts.append(fact)
                self.stats["admissions_count"] += 1
            elif fact["type"] == "denial":
                self.denied_facts.append(fact)
                self.stats["denials_count"] += 1

        # 证据提取
        evidence = self._extract_evidence_mentions(answer, round_num)
        self.evidence_mentioned.extend(evidence)

        # 矛盾检测
        contradictions = self._detect_contradictions(answer, round_num)
        if contradictions:
            self.contradictions.extend(contradictions)
            self.stats["contradictions_count"] += len(contradictions)

        # 嫌疑人特征提取
        self._extract_suspect_features(answer, round_num)

        # 时间线提取
        self._extract_timeline_info(answer, round_num)

        # 实体提取
        self._extract_entities(answer, round_num)

    def _analyze_conversation(self, question: str, answer: str, round_num: int) -> Dict:
        """深度分析对话内容"""
        analysis = {
            "question_type": self._classify_question(question),
            "answer_style": self._classify_answer_style(answer),
            "compliance_level": self._assess_compliance(answer),
            "emotional_tone": self._detect_emotional_tone(answer),
            "credibility_score": self._assess_credibility(answer, round_num),
            "key_points": self._extract_key_points(answer),
            "follow_up_needed": self._identify_follow_up_needs(question, answer)
        }
        return analysis

    # ============ 文本处理辅助方法 ============

    def _normalize_text(self, text: str) -> str:
        """文本标准化"""
        # 移除标点、转换为小写、去除停用词
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        # 中文停用词（简化版）
        stop_words = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '们',
                      '就', '都', '也', '还', '和', '与', '或', '及', '等'}

        filtered_words = [w for w in words if w not in stop_words]

        # 排序以确保顺序不影响比较
        return ' '.join(sorted(filtered_words[:7]))  # 取前7个关键词

    def _classify_question(self, question: str) -> str:
        """问题分类"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['吗', '是不是', '有没有', '是否']):
            return "是非问"
        elif any(word in question_lower for word in ['什么', '哪', '哪里', '何时', '为什么']):
            return "特指问"
        elif any(word in question_lower for word in ['解释', '说明', '详细', '具体']):
            return "解释问"
        elif any(word in question_lower for word in ['记得', '回忆', '想起']):
            return "回忆问"
        elif any(word in question_lower for word in ['证据', '监控', '照片', '录音']):
            return "证据问"
        else:
            return "一般问"

    def _classify_answer_style(self, answer: str) -> str:
        """回答风格分类"""
        answer_length = len(answer)

        if answer_length < 20:
            return "简短回避"
        elif answer_length > 100:
            return "冗长模糊"
        elif "不记得" in answer or "没印象" in answer:
            return "失忆推脱"
        elif "我承认" in answer or "我交代" in answer:
            return "部分承认"
        elif any(word in answer for word in ["没有", "不是", "否认"]):
            return "直接否认"
        elif "可能是" in answer or "也许" in answer:
            return "模糊不确定"
        else:
            return "正常回答"

    # ============ 事实提取方法 ============

    def _extract_facts_from_answer(self, answer: str, round_num: int) -> List[Dict]:
        """从回答中提取事实"""
        facts = []

        # 承认类事实
        admission_patterns = [
            (r'(我|确实|承认)(.*?)(是|有|在|到)', 'admission'),
            (r'(没错|对的|正确|是的)', 'admission'),
            (r'(我交代|我承认|我坦白)', 'strong_admission')
        ]

        # 否认类事实
        denial_patterns = [
            (r'(没有|不是|否认|从未|没)(.*?)(过|过)', 'denial'),
            (r'(不可能|绝对不会|绝无此事)', 'strong_denial')
        ]

        # 检查承认模式
        for pattern, fact_type in admission_patterns:
            matches = re.finditer(pattern, answer)
            for match in matches:
                fact_text = match.group()
                fact = {
                    "type": "admission",
                    "subtype": fact_type,
                    "fact": fact_text,
                    "context": answer[max(0, match.start() - 30):match.end() + 30],
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.8 if fact_type == "strong_admission" else 0.6
                }
                facts.append(fact)

        # 检查否认模式
        for pattern, fact_type in denial_patterns:
            matches = re.finditer(pattern, answer)
            for match in matches:
                fact_text = match.group()
                fact = {
                    "type": "denial",
                    "subtype": fact_type,
                    "fact": fact_text,
                    "context": answer[max(0, match.start() - 30):match.end() + 30],
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.9 if fact_type == "strong_denial" else 0.7
                }
                facts.append(fact)

        return facts

    # ============ 矛盾检测方法 ============

    def _detect_contradictions(self, answer: str, round_num: int) -> List[Dict]:
        """检测矛盾"""
        contradictions = []

        if len(self.conversation_history) < 2:
            return contradictions

        # 获取最近几轮的历史
        recent_history = self.conversation_history[-3:] if len(
            self.conversation_history) >= 3 else self.conversation_history

        for hist_record in recent_history:
            if hist_record["round"] == round_num:
                continue

            prev_answer = hist_record["answer"]

            # 简单关键词矛盾检测
            contradictions.extend(
                self._check_keyword_contradictions(answer, prev_answer, round_num, hist_record["round"]))

            # 时间矛盾检测
            contradictions.extend(self._check_time_contradictions(answer, prev_answer, round_num, hist_record["round"]))

            # 数量矛盾检测
            contradictions.extend(
                self._check_quantity_contradictions(answer, prev_answer, round_num, hist_record["round"]))

        return contradictions

    def _check_keyword_contradictions(self, curr_answer: str, prev_answer: str, curr_round: int, prev_round: int) -> \
    List[Dict]:
        """关键词矛盾检测"""
        contradictions = []

        # 定义矛盾关键词对
        contradiction_pairs = [
            ("是", "不是"), ("有", "没有"), ("在", "不在"),
            ("见过", "没见过"), ("认识", "不认识"), ("知道", "不知道")
        ]

        for pos_word, neg_word in contradiction_pairs:
            if pos_word in curr_answer and neg_word in prev_answer:
                contradiction = {
                    "type": "关键词矛盾",
                    "current_round": curr_round,
                    "previous_round": prev_round,
                    "current_word": pos_word,
                    "previous_word": neg_word,
                    "description": f"第{prev_round}轮说'{neg_word}'，第{curr_round}轮说'{pos_word}'",
                    "severity": "high",
                    "timestamp": datetime.now().isoformat()
                }
                contradictions.append(contradiction)

        return contradictions

    # ============ 特征提取方法 ============

    def _extract_suspect_features(self, answer: str, round_num: int) -> None:
        """提取嫌疑人特征"""
        # 家庭情况
        family_keywords = ["妻子", "女儿", "儿子", "父母", "家庭", "家人", "孩子", "老婆"]
        for keyword in family_keywords:
            if keyword in answer:
                self.suspect_profile["家庭情况"].append({
                    "keyword": keyword,
                    "context": answer,
                    "round": round_num
                })

        # 健康状况
        health_keywords = ["胃", "身体", "健康", "生病", "医院", "医生", "休息", "睡眠"]
        for keyword in health_keywords:
            if keyword in answer:
                self.suspect_profile["健康状况"].append({
                    "keyword": keyword,
                    "context": answer,
                    "round": round_num
                })

        # 工作习惯
        work_keywords = ["工作", "单位", "项目", "加班", "会议", "文件", "汇报"]
        for keyword in work_keywords:
            if keyword in answer:
                self.suspect_profile["工作习惯"].append({
                    "keyword": keyword,
                    "context": answer,
                    "round": round_num
                })

    # ============ 公开API方法 ============

    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        summary = []

        summary.append(f"📊 审讯记忆摘要 (共{self.stats['total_rounds']}轮)")
        summary.append("=" * 50)

        # 对话统计
        summary.append(f"📝 对话统计:")
        summary.append(f"  总轮次: {self.stats['total_rounds']}")
        summary.append(f"  承认事实: {self.stats['admissions_count']}")
        summary.append(f"  否认事实: {self.stats['denials_count']}")
        summary.append(f"  发现矛盾: {self.stats['contradictions_count']}")

        # 当前状态
        summary.append(f"\n🎯 当前状态:")
        summary.append(f"  审讯阶段: {self.interrogation_phase}")
        summary.append(f"  施压级别: {self.pressure_level}/10")

        # 最近对话
        if self.conversation_history:
            summary.append(f"\n💬 最近对话:")
            for record in self.conversation_history[-3:]:
                summary.append(f"  第{record['round']}轮 [{record['phase']}]:")
                summary.append(f"    问: {record['question'][:50]}...")
                summary.append(f"    答: {record['answer'][:50]}...")

        # 关键事实
        if self.confirmed_facts:
            summary.append(f"\n✅ 已确认事实:")
            for fact in self.confirmed_facts[-3:]:
                summary.append(f"  - {fact['fact'][:60]}...")

        # 发现矛盾
        if self.contradictions:
            summary.append(f"\n⚠️  发现矛盾:")
            for contr in self.contradictions[-2:]:
                summary.append(f"  - {contr['description'][:80]}...")

        # 待追问问题
        if self.pending_inquiries:
            summary.append(f"\n❓ 待追问问题:")
            for inquiry in self.pending_inquiries[-3:]:
                summary.append(f"  - {inquiry[:60]}...")

        summary.append("=" * 50)

        return "\n".join(summary)

    def suggest_next_questions(self, count: int = 3) -> List[str]:
        """建议下一个问题"""
        suggestions = []

        # 基于矛盾点的追问
        if self.contradictions:
            latest = self.contradictions[-1]
            suggestions.append(
                f"关于你在第{latest['previous_round']}轮和第{latest['current_round']}轮回答中的矛盾（{latest['description'][:30]}...），请解释一下。")

        # 基于已承认事实追问细节
        if self.confirmed_facts:
            latest_fact = self.confirmed_facts[-1]
            suggestions.append(f"你刚才承认了'{latest_fact['fact'][:30]}...'，能否提供更多细节？")

        # 基于审讯阶段的策略性问题
        if self.interrogation_phase == "建立关系":
            suggestions.extend([
                "能否详细谈谈你的家庭情况？",
                "你在单位主要负责哪些工作？"
            ])
        elif self.interrogation_phase == "证据突袭":
            suggestions.extend([
                "关于监控录像显示的情况，你有什么要说的？",
                "那笔50万的资金往来，到底是什么性质？"
            ])

        # 基于嫌疑人特征的问题
        if "健康状况" in self.suspect_profile and self.suspect_profile["健康状况"]:
            suggestions.append("你刚才提到健康问题，具体是什么情况？")

        return suggestions[:count]

    def update_interrogation_phase(self, round_number: int, total_rounds: int) -> None:
        """更新审讯阶段"""
        progress = round_number / total_rounds

        if progress < 0.25:
            self.interrogation_phase = "建立关系"
            self.pressure_level = 2
        elif progress < 0.5:
            self.interrogation_phase = "试探提问"
            self.pressure_level = 5
        elif progress < 0.75:
            self.interrogation_phase = "证据突袭"
            self.pressure_level = 8
        else:
            self.interrogation_phase = "极限施压"
            self.pressure_level = 10

    def add_pending_inquiry(self, inquiry: str) -> None:
        """添加待追问问题"""
        if inquiry and inquiry not in self.pending_inquiries:
            self.pending_inquiries.append(inquiry)

    def get_conversation_context(self, num_rounds: int = 5) -> str:
        """获取对话上下文"""
        if not self.conversation_history:
            return "暂无对话历史"

        recent = self.conversation_history[-num_rounds:]
        context = []

        for record in recent:
            context.append(f"第{record['round']}轮 [{record['phase']}]")
            context.append(f"问：{record['question']}")
            context.append(f"答：{record['answer'][:100]}...")
            context.append("-" * 40)

        return "\n".join(context)

    def save_to_file(self, filename: str) -> None:
        """保存记忆到文件"""
        memory_data = {
            "suspect_name": self.suspect_name,
            "conversation_history": self.conversation_history,
            "confirmed_facts": self.confirmed_facts,
            "denied_facts": self.denied_facts,
            "contradictions": self.contradictions,
            "suspect_profile": self.suspect_profile,
            "interrogation_phase": self.interrogation_phase,
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

        print(f"[记忆系统] 记忆已保存到 {filename}")

    def load_from_file(self, filename: str) -> bool:
        """从文件加载记忆"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            # 恢复所有记忆
            self.suspect_name = memory_data.get("suspect_name", self.suspect_name)
            self.conversation_history = memory_data.get("conversation_history", [])
            self.confirmed_facts = memory_data.get("confirmed_facts", [])
            self.denied_facts = memory_data.get("denied_facts", [])
            self.contradictions = memory_data.get("contradictions", [])
            self.suspect_profile = memory_data.get("suspect_profile", self.suspect_profile)
            self.interrogation_phase = memory_data.get("interrogation_phase", "建立关系")
            self.stats = memory_data.get("stats", self.stats)

            # 重建asked_questions集合
            self.asked_questions.clear()
            for record in self.conversation_history:
                self.asked_questions.add(self._normalize_text(record["question"]))

            print(f"[记忆系统] 记忆已从 {filename} 加载")
            return True

        except Exception as e:
            print(f"[记忆系统] 加载失败: {e}")
            return False

#conversation_history - 完整对话记录
#confirmed_facts - 已确认事实
#contradictions - 矛盾点检测
#suspect_profile - 嫌疑人特征档案
#timeline_facts - 时间线记忆
#stats - 审讯统计
#add_conversation() - 添加完整对话记录
#get_memory_summary() - 获取详细摘要
#suggest_next_questions() - 智能问题建议
#save_to_file() / load_from_file() - 记忆持久化
#profile - 查看嫌疑人特征
#stats - 查看审讯统计
#phase - 查看审讯阶段
#load - 加载之前保存的审讯