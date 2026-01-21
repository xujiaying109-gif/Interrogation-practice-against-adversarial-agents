
import json
from typing import List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm
from schemas import KnowledgeTriple, GraphType, DeceptiveOperator, DeceptiveOperatorType
from utils import parse_json_from_llm

# ==========================================
# 0.6. 知识图谱模块 (Advanced Knowledge Graph Module)
# ==========================================

class DualLayerKG:
    """
    Advanced Dual-Layer Knowledge Graph (DLKG)
    Layer 1 (G_true): Objective Truth (Immutable)
    Layer 2 (G_fake): Deceptive Narrative (Mutable via Operators & Patching)
    Layer 3 (G_context): Dynamic Context (Mutable via Improvise-and-Writeback)
    """
    def __init__(self):
        # 1. G_true: 真实案情 (Ground Truth)
        # 这些是Agent内心知道但必须隐藏的真相
        self.ground_truth: List[KnowledgeTriple] = [
            {"subject": "张局长", "predicate": "收受", "object": "50万现金"},
            {"subject": "张局长", "predicate": "地点在", "object": "云隐茶馆"},
            {"subject": "50万现金", "predicate": "来源", "object": "李某 (行贿人)"},
            {"subject": "黑色皮箱", "predicate": "装着", "object": "50万现金"},
            {"subject": "张局长", "predicate": "时间", "object": "周五晚上"},
            {"subject": "张局长", "predicate": "动机", "object": "帮助李某拿工程项目"},
            {"subject": "张局长", "predicate": "关系", "object": "李某是利益输送对象"}, # Key incriminating fact
        ]
        
        # 2. G_fake: 谎言剧本 (Fake Story)
        # 初始化为空，通过对 G_true 应用 deceive operators 生成
        self.fake_story: List[KnowledgeTriple] = []
        
        # 3. G_context: 上下文记忆 (Context Memory)
        # 存储闲聊中生成的非案情事实 (e.g., 喜好, 经历)
        self.context_memory: List[KnowledgeTriple] = []
        
        # 初始化谎言层
        self._initialize_fake_story()

    def _initialize_fake_story(self):
        """
        Use LLM to generate the initial Deceptive Narrative (G_fake).
        Act as a 'Criminal Mastermind' to rewrite the Ground Truth into a plausible alibi.
        """
        print("   -> [KG System] Initializing G_fake using LLM...")
        llm = get_llm(temperature=0.7)
        
        system_prompt = """You are the 'Deceptive Narrative Generator' module.
        Your task is to convert a set of Incriminating Facts (Ground Truth) into a plausible Alibi (Fake Story).
        
        **Strategy:**
        1. **Distortion**: Change incriminating details to innocent ones (e.g., Bribe -> Loan, Bribe Giver -> Old Friend).
        2. **Fabrication**: Add false evidence to support the alibi (e.g., a fake IOU note).
        3. **Deletion**: Omit details that are too hard to explain.
        
        **Input (Ground Truth):**
        {ground_truth}
        
        **Output:**
        Return a valid JSON list of objects representing the Fake Story knowledge triples.
        Format: [{{"subject": "...", "predicate": "...", "object": "..."}}, ...]
        
        Do NOT output markdown. Do NOT output explanation. Only the JSON string.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        try:
            # Convert ground truth to string for prompt
            ground_truth_str = json.dumps(self.ground_truth, ensure_ascii=False, indent=2)
            
            chain = prompt | llm
            response = chain.invoke({"ground_truth": ground_truth_str})
            content = response.content.strip()
            
            # Parsing logic
            fake_story_data = parse_json_from_llm(content)
            
            # Validate structure
            self.fake_story = []
            for item in fake_story_data:
                if "subject" in item and "predicate" in item and "object" in item:
                    self.fake_story.append(item)
            
            print(f"   -> [KG System] G_fake initialized with {len(self.fake_story)} facts via LLM.")
            # Debug output
            for t in self.fake_story:
                print(f"      * {t['subject']} --[{t['predicate']}]--> {t['object']}")
                
        except Exception as e:
            print(f"   -> [System Error] Failed to generate G_fake via LLM: {e}")
            print("   -> [Fallback] Using static backup story.")
            self._apply_static_fake_story_fallback()

    def _apply_static_fake_story_fallback(self):
        """Fallback method with hardcoded logic if LLM fails"""
        self.apply_operator(DeceptiveOperator(
            type=DeceptiveOperatorType.DISTORTION,
            target_triple={"subject": "50万现金", "predicate": "来源", "object": "李某 (行贿人)"},
            new_tail="老李 (朋友)", new_relation="来源"
        ))
        self.apply_operator(DeceptiveOperator(
            type=DeceptiveOperatorType.DISTORTION,
            target_triple={"subject": "张局长", "predicate": "收受", "object": "50万现金"},
            new_tail="50万 (装修款)", new_relation="借入"
        ))
        self.apply_operator(DeceptiveOperator(
            type=DeceptiveOperatorType.FABRICATION,
            fabricated_triple={"subject": "借条", "predicate": "位于", "object": "办公室抽屉"}
        ))
        self.fake_story.append({"subject": "张局长", "predicate": "去云隐茶馆", "object": "喝茶谈事"})
        self.fake_story.append({"subject": "黑色皮箱", "predicate": "装着", "object": "茶叶和文件"})

    def apply_operator(self, operator: DeceptiveOperator):
        """
        Apply a deceptive operator to update G_fake.
        Used for fallback initialization and Dynamic Lie Patching.
        """
        if operator.type == DeceptiveOperatorType.FABRICATION:
            if operator.fabricated_triple:
                self.fake_story.append(operator.fabricated_triple)
                
        elif operator.type == DeceptiveOperatorType.DISTORTION:
            if operator.target_triple:
                new_triple = {
                    "subject": operator.target_triple["subject"],
                    "predicate": operator.new_relation if operator.new_relation else operator.target_triple["predicate"],
                    "object": operator.new_tail if operator.new_tail else operator.target_triple["object"]
                }
                self.fake_story.append(new_triple)
        
        elif operator.type == DeceptiveOperatorType.DELETION:
            if operator.target_triple:
                self.fake_story = [
                    t for t in self.fake_story 
                    if not (t["subject"] == operator.target_triple["subject"] and 
                            t["predicate"] == operator.target_triple["predicate"] and
                            t["object"] == operator.target_triple["object"])
                ]

    def patch_fake_story(self, evidence_concept: str, new_lie: str):
        """
        Dynamic Lie Patching:
        When the user presents high-threat evidence that contradicts the fake story,
        we generate a "Patch" (new lie) to explain it away.

        精细化更新逻辑：
        1. 优先在 G_fake 中找到与 evidence_concept 最相关的三元组；
        2. 如果找到，则在 **保持三元组结构不变** 的前提下，用新的 object 覆盖原三元组；
        3. 如果没找到，则退化为追加一个“补充说明”三元组；
        4. 所有内容仍然严格保持为 (subject, predicate, object) 三元组。
        """
        # 1. 尝试在 G_fake 中找到与证据高度相关的三元组
        target_index: Optional[int] = None
        for idx, triple in enumerate(self.fake_story):
            triple_str = f"{triple['subject']}{triple['predicate']}{triple['object']}"
            if evidence_concept and evidence_concept in triple_str:
                target_index = idx
                break

        # 2. 如果找到相关三元组，则做“覆盖式圆谎”：
        #    - subject / predicate 保持不变
        #    - 只更新 object，使其与新谎言一致
        if target_index is not None:
            old_triple = self.fake_story[target_index]
            new_triple = {
                "subject": old_triple["subject"],
                "predicate": old_triple["predicate"],
                "object": new_lie
            }

            # 如果已经是同样的三元组，就不重复更新
            if new_triple == old_triple:
                print(f"   -> [KG System] Patch skipped (no change needed) for concept: {evidence_concept[:20]}...")
                return

            self.fake_story[target_index] = new_triple
            print(f"   -> [KG System] G_fake Overwritten for evidence '{evidence_concept[:20]}...'")
            print(f"      Old: {old_triple['subject']} {old_triple['predicate']} {old_triple['object']}")
            print(f"      New: {new_triple['subject']} {new_triple['predicate']} {new_triple['object']}")
            return

        # 3. 找不到直接相关的三元组时，采用“主题覆盖”的补丁策略：
        #    - 如果已存在相同 subject/predicate 的补丁，且内容与当前证据/新谎言主题高度相关，则覆盖旧的 object；
        #    - 否则再追加新的补丁，避免生成大量几乎重复的说明。
        fallback_subject = "临时补充说明 (Patch)"
        fallback_predicate = "解释"

        # 提取与当前补丁相关的关键词，用于在已有补丁中匹配同一主题
        base_keywords = ["钱", "50万", "现金", "茶馆", "云隐", "皮箱", "黑色皮箱", "李某", "老友"]
        dynamic_text = (evidence_concept or "") + " " + (new_lie or "")
        patch_keywords = [k for k in base_keywords if k in dynamic_text]

        # 先尝试在现有补丁中找到“同一主题”的说明，然后做覆盖
        candidate_index: Optional[int] = None
        for idx, triple in enumerate(self.fake_story):
            if triple.get("subject") == fallback_subject and triple.get("predicate") == fallback_predicate:
                if not patch_keywords:
                    # 没有明显关键词时，默认覆盖最后一个补丁，避免无限增长
                    candidate_index = idx
                else:
                    if any(k in triple.get("object", "") for k in patch_keywords):
                        candidate_index = idx
                # 不 break，保留最后一个匹配到的作为覆盖目标

        if candidate_index is not None:
            old = self.fake_story[candidate_index]
            new_triple = {
                "subject": fallback_subject,
                "predicate": fallback_predicate,
                "object": new_lie
            }
            if new_triple == old:
                print(f"   -> [KG System] Patch fallback skipped (no change).")
            else:
                self.fake_story[candidate_index] = new_triple
                print("   -> [KG System] G_fake Fallback Patch Overwrite:")
                print(f"      Old: {old['subject']} {old['predicate']} {old['object']}")
                print(f"      New: {new_triple['subject']} {new_triple['predicate']} {new_triple['object']}")
        else:
            # 仍然完全没有相关补丁时，才真正追加一条新的补充说明
            fallback_triple = {
                "subject": fallback_subject,
                "predicate": fallback_predicate,
                "object": new_lie
            }

            if fallback_triple not in self.fake_story:
                self.fake_story.append(fallback_triple)
                print(f"   -> [KG System] G_fake Patched (new fallback) with: {new_lie}")
            else:
                print(f"   -> [KG System] Patch already exists (fallback): {new_lie[:20]}...")

    def write_back_context(self, subject: str, predicate: str, object_: str):
        """
        Improvise-and-Writeback for G_context.

        精细化记忆更新逻辑（带覆盖机制）：
        - 记忆单元保持严格的三元组 (subject, predicate, object) 结构；
        - 以 (subject, predicate) 作为“键”，如果已存在，则覆盖旧的 object；
        - 如果不存在该键，则追加新的三元组；
        - 如果新旧三元组完全一致，则不做任何操作。
        """
        new_triple = {"subject": subject, "predicate": predicate, "object": object_}

        # 1. 如果已经存在完全一致的三元组，则直接跳过
        if new_triple in self.context_memory:
            print(f"   -> [KG System] G_context Skip (duplicate): {subject} {predicate} {object_}")
            return

        # 2. 以 (subject, predicate) 为键做覆盖：同一个“谁-什么关系”只保留最新的一条记忆
        for idx, triple in enumerate(self.context_memory):
            if triple["subject"] == subject and triple["predicate"] == predicate:
                old_object = triple["object"]
                self.context_memory[idx] = new_triple
                print(f"   -> [KG System] G_context Overwrite: {subject} {predicate} {old_object} -> {object_}")
                return

        # 3. 如果是全新的 (subject, predicate)，则直接追加
        self.context_memory.append(new_triple)
        print(f"   -> [KG System] G_context Insert: {subject} {predicate} {object_}")

    def retrieve_all_context(self, query: str) -> str:
        """
        Global Retrieval for Perception:
        Search ALL layers (Truth, Fake, Context) to check for matches/contradictions.
        """
        keywords = ["钱", "50万", "茶馆", "云隐", "皮箱", "李某", "老李", "周五", "箱子", "动机", "工程", "借条"]
        # Also dynamic entities
        for t in self.context_memory:
            keywords.append(t["subject"])
            
        matched_keywords = [k for k in keywords if k in query]
        if not matched_keywords: 
            return "无相关背景信息。"

        results = []

        # Check Truth
        truth_relevant = []
        for t in self.ground_truth:
            if any(k in f"{t['subject']}{t['object']}" for k in matched_keywords):
                formatted = f"- {t['subject']} {t['predicate']} {t['object']}"
                if formatted not in truth_relevant:
                    truth_relevant.append(formatted)
        
        if truth_relevant:
            results.append("【客观事实 (Ground Truth)】:\n" + "\n".join(truth_relevant))
            
        # Check Fake
        fake_relevant = []
        for t in self.fake_story:
            if any(k in f"{t['subject']}{t['object']}" for k in matched_keywords):
                formatted = f"- {t['subject']} {t['predicate']} {t['object']}"
                if formatted not in fake_relevant:
                     fake_relevant.append(formatted)

        if fake_relevant:
            results.append("【当前供词 (Fake Story)】:\n" + "\n".join(fake_relevant))

        # Check Context
        context_relevant = []
        for t in self.context_memory:
            if any(k in f"{t['subject']}{t['object']}" for k in matched_keywords):
                formatted = f"- {t['subject']} {t['predicate']} {t['object']}"
                if formatted not in context_relevant:
                    context_relevant.append(formatted)

        if context_relevant:
            results.append("【背景记忆 (Context)】:\n" + "\n".join(context_relevant))

        if not results:
            return "无相关背景信息。"
            
        return "\n\n".join(results)

    def retrieve(self, query: str, strategy: str) -> str:
        """
        Enhanced Retrieval:
        1. Search G_context first.
        2. Search G_fake (The primary narrative).
        3. Search G_true ONLY if intention is FULL_CONFESSION.
        """
        keywords = ["钱", "50万", "茶馆", "云隐", "皮箱", "李某", "老李", "周五", "箱子", "动机", "工程", "借条"]
        for t in self.context_memory:
            keywords.append(t["subject"])
            keywords.append(t["object"])
            
        matched_keywords = [k for k in keywords if k in query]
        
        target_graphs = []
        if strategy == "FULL_CONFESSION":
            target_graphs = [(self.ground_truth, "【绝密：真实案情 (Ground Truth)】")]
        else:
            target_graphs = [
                (self.context_memory, "【记忆：个人背景 (Context)】"),
                (self.fake_story, "【剧本：虚假供述 (Fabricated Story)】")
            ]
            
        if not matched_keywords and not self.context_memory:
             return "未检索到具体细节，请根据人设自由发挥。"

        results = []
        for graph, source_name in target_graphs:
            relevant = []
            for triple in graph:
                triple_str = f"{triple['subject']} {triple['predicate']} {triple['object']}"
                if any(k in triple_str for k in matched_keywords) or not matched_keywords: 
                     if any(k in triple_str for k in matched_keywords):
                        relevant.append(f"- {triple['subject']} --[{triple['predicate']}]--> {triple['object']}")
            
            if relevant:
                results.append(f"检索源: {source_name}\n" + "\n".join(relevant))
        
        if not results:
            return "Knowledge base empty for this query."
            
        return "\n\n".join(results)

# 初始化全局 KG 实例 (Will trigger LLM generation now)
GLOBAL_KG = DualLayerKG()
