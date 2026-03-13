import json
from enum import Enum
from typing import Dict, List, Any

class NodeStatus(Enum):
    HIDDEN = "hidden"
    EXPOSED = "exposed"
    BROKEN = "broken"  # 防线被击穿，已承认


class FourDGraph:
    """
    四维图谱数据结构与状态机管理引擎
    人 (persons) - 权 (powers) - 钱 (monies) - 事 (matters)
    """
    def __init__(self, case_json: Dict[str, Any]):
        self.nodes = {
            "persons": {},
            "powers": {},
            "monies": {},
            "matters": {}
        }
        self.edges = []
        self._load_data(case_json)

    def _load_data(self, data: Dict[str, Any]):
        nodes_data = data.get("nodes", {})
        
        for p in nodes_data.get("persons", []):
            self.nodes["persons"][p["id"]] = p
            if "status" not in self.nodes["persons"][p["id"]]:
                self.nodes["persons"][p["id"]]["status"] = NodeStatus.HIDDEN.value
                
        for pw in nodes_data.get("powers", []):
            self.nodes["powers"][pw["id"]] = pw
            if "status" not in self.nodes["powers"][pw["id"]]:
                self.nodes["powers"][pw["id"]]["status"] = NodeStatus.HIDDEN.value
                
        for m in nodes_data.get("monies", []):
            self.nodes["monies"][m["id"]] = m
            if "status" not in self.nodes["monies"][m["id"]]:
                self.nodes["monies"][m["id"]]["status"] = NodeStatus.HIDDEN.value
                
        for mat in nodes_data.get("matters", []):
            self.nodes["matters"][mat["id"]] = mat
            if "status" not in self.nodes["matters"][mat["id"]]:
                self.nodes["matters"][mat["id"]]["status"] = NodeStatus.HIDDEN.value

        self.edges = data.get("edges", [])

    def evaluate_triggers(self, presented_evidence: List[str]) -> List[Dict[str, Any]]:
        """
        根据审讯官目前累计出示的证据ID列表，评估是否有隐秘节点的防御被击穿
        :param presented_evidence: 当前已出示并被嫌疑人感知的证据 ID 列表 (如 ["ev_bank_transfer_li"])
        :return: 本次刚刚被击穿的节点列表
        """
        presented_set = set(presented_evidence)
        newly_broken = []
        
        for category, items in self.nodes.items():
            for node_id, node in items.items():
                if node.get("status") == NodeStatus.BROKEN.value:
                    continue  # 已经承认的节点不再处理
                
                conditions = node.get("break_conditions", {})
                required_evidence = conditions.get("required_evidence", [])
                
                # 如果该节点有触发条件，且该条件是被出示证据的子集，则击穿防线
                if required_evidence and set(required_evidence).issubset(presented_set):
                    node["status"] = NodeStatus.BROKEN.value
                    newly_broken.append(node)
                    
        return newly_broken

    def get_fallback_strategy_for_node(self, node_id: str, category: str) -> str:
        """获取某个节点在还没被完全击穿时，应该使用的抵抗/欺诈算子名称"""
        node = self.nodes.get(category, {}).get(node_id, {})
        return node.get("break_conditions", {}).get("fallback_strategy", "直接否认")

    def dump_kg_context(self) -> str:
        """序列化当前的图谱状态，用于放入 LLM prompt"""
        context = "【当前案件四维图谱状态】\n"
        
        for cat_name, cat_nodes in self.nodes.items():
            context += f"--- {cat_name.upper()} ---\n"
            for n_id, n_data in cat_nodes.items():
                status = n_data.get("status")
                name_or_desc = n_data.get("name", n_data.get("desc", n_data.get("amount", n_id)))
                context += f"- [{status}] {name_or_desc}\n"
                
        return context
