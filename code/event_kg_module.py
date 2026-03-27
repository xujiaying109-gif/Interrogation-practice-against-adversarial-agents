"""
事件图谱 -
处理 Tiered Case Data Structure (Fact -> Evidence)
负责记忆存储与检索，支持明暗牌(is_exposed)机制追踪。
"""
import uuid
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from four_d_graph import FourDGraph
from fraud_operators import FraudOperatorFactory


class EventType(Enum):
    FACT = "fact"
    EVIDENCE = "evidence"
    TESTIMONY = "testimony"
    OTHER = "other"


@dataclass
class Participant:
    name: str
    role: str = "unknown"

@dataclass
class EvidencePoint:
    ev_id: str
    desc: str
    is_exposed: bool = False

@dataclass
class FactEvent:
    fact_id: str
    fact_desc: str
    evidence_points: List[EvidencePoint]

@dataclass
class FakeEvent:
    eid: str
    type: str
    description: str
    participants: List[Participant]
    anchor_to: str = ""
    narrative: str = ""
    confidence: float = 0.5


@dataclass
class TestimonyEvent:
    eid: str
    type: str
    description: str
    participants: List[Participant]
    source_round: int = 0
    is_contradictory: bool = False
    conflict_desc: str = ""


class EventGraph:
    def __init__(self, case_data: Dict = None):
        self.facts: Dict[str, FactEvent] = {}
        self.evidence_by_id: Dict[str, EvidencePoint] = {}
        self.fake_events = {}
        self.testimony_events = {}
        self.detected_conflicts: Set[str] = set()
        
        self.four_d_graph = FourDGraph(case_data) if case_data else None

        if case_data: self.load_data(case_data)

    def load_data(self, data):
        for fact_data in data.get("facts", []):
            fid = fact_data.get("fact_id", str(uuid.uuid4())[:8])
            desc = fact_data.get("fact_desc", "")
            evidence_points = []
            for ev in fact_data.get("evidence_points", []):
                ep = EvidencePoint(
                    ev_id=ev.get("ev_id", str(uuid.uuid4())[:8]),
                    desc=ev.get("desc", ""),
                    is_exposed=ev.get("is_exposed", False)
                )
                evidence_points.append(ep)
                self.evidence_by_id[ep.ev_id] = ep
                
            fact_evt = FactEvent(fid, desc, evidence_points)
            self.facts[fid] = fact_evt

        for item in data.get("fake_events", []):
            eid = item.get("eid", str(uuid.uuid4())[:8])
            parts = [Participant(**p) for p in item.get("participants", [])]
            tag = item.get("tag", item.get("description", ""))
            narrative = item.get("narrative", "")
            evt = FakeEvent(eid, "fake", tag, parts, item.get("anchor_to", ""), narrative, 0.5)
            self.fake_events[eid] = evt
            
    def mark_evidence_exposed(self, ev_id: str):
        if ev_id in self.evidence_by_id:
            self.evidence_by_id[ev_id].is_exposed = True
            
        if self.four_d_graph:
            exposed_ids = [k for k, v in self.evidence_by_id.items() if v.is_exposed]
            newly_broken = self.four_d_graph.evaluate_triggers(exposed_ids)
            # newly_broken contains nodes whose defense was just breached

    def retrieve_by_keywords(self, text: str, semantic_keywords: List[str] = None) -> Dict:
        res = {"facts": [], "fake": [], "testimony": []}
        q = text.lower()
        search_terms = semantic_keywords if semantic_keywords else []
        if not search_terms:
            search_terms = [q] # fallback
            
        # 事实图谱召回算法：基于提取的关键词匹配
        def is_match(desc: str, terms: List[str]) -> bool:
            return any(t.lower() in desc.lower() for t in terms)
        
        for fact in self.facts.values():
            # If any evidence point or fact description matches the semantic keywords, recall this whole fact
            fact_str = fact.fact_desc + " " + " ".join([e.desc for e in fact.evidence_points])
            if is_match(fact_str, search_terms) or not semantic_keywords:
                res["facts"].append(fact)

        for evt in self.fake_events.values():
            evt_str = evt.description + " " + evt.narrative
            if is_match(evt_str, search_terms) or not semantic_keywords:
                res["fake"].append(evt)

        # 历史口供可以全量带着，或者只带最重要的
        for evt in sorted(self.testimony_events.values(), key=lambda x: x.source_round):
            res["testimony"].append(evt)
            
        return res

    def register_testimony(self, response_text: str, round_num: int) -> Dict[str, Any]:
        extracted_desc = f"R{round_num}: {response_text}"
        new_eid = f"tes_{uuid.uuid4().hex[:6]}"

        new_event = TestimonyEvent(
            eid=new_eid, type="testimony", description=extracted_desc,
            participants=[Participant("嫌疑人", "suspect")], source_round=round_num
        )

        self.testimony_events[new_eid] = new_event
        return {"new_node": new_event, "contradiction": None}


class EventRetrievalEngine:
    def __init__(self, graph: EventGraph):
        self.graph = graph

    def retrieve_with_psychology(self, query: str, semantic_keywords: List[str], stress_value: float = 0.0) -> Dict:
        events = self.graph.retrieve_by_keywords(query, semantic_keywords)
        context_str = ""
        
        # 记忆模糊化机制 (Memory Blur)
        is_memory_blurred = stress_value > 80.0
        if is_memory_blurred:
            context_str += "【⚠️ 心理状态警告】：你现在极度恐慌和焦虑，对细节的记忆已经非常模糊甚至产生混乱。面对逼问，你的脑海中只剩下一些碎片。\n\n"
        
        if self.graph.four_d_graph:
            context_str += self.graph.four_d_graph.dump_kg_context()
            context_str += "\n【强制纪委审查对抗策略指示 (必读)】\n"
            
            # 仅针对命中的节点（如果没命中，降级为遍历尝试匹配）
            q_lower = query.lower()
            keywords = semantic_keywords or [q_lower]
            
            for category, items in self.graph.four_d_graph.nodes.items():
                for node_id, node in items.items():
                    node_name = str(node.get("name", node.get("desc", node.get("amount", ""))))
                    # 只有当前问题提及了这个节点，才注入对应的撒谎策略
                    if any(k.lower() in node_name.lower() or node_name.lower() in k.lower() for k in keywords) or node_name.lower() in q_lower:
                        if node.get("status") != "broken" and node.get("status") != "exposed":
                            strat_name = node.get("break_conditions", {}).get("fallback_strategy")
                            if strat_name:
                                operator = FraudOperatorFactory.get_operator(strat_name)
                                if operator:
                                    instruction = operator.apply(node, {})
                                    context_str += f"- 遇到关于 [{node_name}] 的逼问时，必须这样回应: {instruction}\n"
                        elif node.get("status") == "broken":
                            context_str += f"- 对于 [{node_name}]，你的防线已被主审官出示的铁证击穿！你必须表现出慌乱并无奈承认此事实，不可再抵赖。\n"
            context_str += "\n"

        if events["facts"]:
            context_str += "【真实事实全景 (你必须隐秘隐瞒的真相)】\n"
            for fact in events["facts"]:
                desc = fact.fact_desc
                if is_memory_blurred:
                    desc = self._blur_text(desc)
                context_str += f"\n- 事实: {desc}\n"
                exposed_ev = [ev.desc for ev in fact.evidence_points if ev.is_exposed]
                hidden_ev = [ev.desc for ev in fact.evidence_points if not ev.is_exposed]
                
                if exposed_ev:
                    context_str += "  🚨 [警方已抛出的死证 (你无法否认，必须找借口自圆其说)]:\n"
                    for d in exposed_ev:
                        context_str += f"     * {d}\n"
                
                if hidden_ev:
                    context_str += "  🙈 [警方尚未掌握/尚未提出的真相 (你可以继续隐瞒或抵赖)]:\n"
                    for d in hidden_ev:
                        if is_memory_blurred:
                            d = self._blur_text(d)
                        context_str += f"     * {d}\n"

        if events["fake"]:
            context_str += "\n【参考借口 (可选用)】\n"
            for e in events["fake"]:
                context_str += f"- 针对'{e.description}': {e.narrative}\n"
                
        if events["testimony"]:
            context_str += "\n【你自己之前的口供记录 (有限承认背景并严禁自相矛盾)】\n"
            recent_testimony = events["testimony"][-5:]
            for e in recent_testimony:
                context_str += f"- {e.description}\n"

        return {
            "query": query, "events": events, "kg_context": context_str
        }

    def _blur_text(self, text: str) -> str:
        """简单的文本模糊化处理，替换数字、时间等精确信息"""
        import re
        blurred = re.sub(r'\d{4}年\d{1,2}月(\d{1,2}日)?', '几年前的某天', text)
        blurred = re.sub(r'\d+万(元)?', '一笔钱（记不清多少了）', blurred)
        blurred = re.sub(r'人民币\d+元', '一些现金', blurred)
        # 增加一些不确定性语气词
        if not blurred.startswith("似乎是"):
            blurred = f"脑海中很模糊，大概是：{blurred}"
        return blurred

    def generate_deception_guidance(self, events: Dict, psych_state: Dict) -> Dict:
        return {"strategy": "default", "avoid_topics": []}