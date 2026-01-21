
from langgraph.graph import StateGraph, END
from schemas import AgentState
from nodes import (
    perception_node, psych_update_node, strategy_node,
    knowledge_retrieval_node, generation_node
)

# ==========================================
# 3. 构建图 (Graph Construction)
# ==========================================

def build_deep_inquisitor_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("perception", perception_node)
    workflow.add_node("psych_update", psych_update_node)
    workflow.add_node("strategy_select", strategy_node)
    workflow.add_node("kg_retrieval", knowledge_retrieval_node)
    workflow.add_node("generate", generation_node)

    # 定义边 (线性流程)
    # Start -> Perception -> Psych -> Strategy -> KG -> Generate -> End
    workflow.set_entry_point("perception")
    workflow.add_edge("perception", "psych_update")
    workflow.add_edge("psych_update", "strategy_select")
    workflow.add_edge("strategy_select", "kg_retrieval")
    workflow.add_edge("kg_retrieval", "generate")
    workflow.add_edge("generate", END)

    # 编译
    app = workflow.compile()
    return app
