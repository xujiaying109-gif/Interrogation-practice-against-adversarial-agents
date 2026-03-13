import pytest
from event_kg_module import EventGraph, EventRetrievalEngine

def test_testimony_retrieval_integration():
    graph = EventGraph({
        "truth_events": [{"description": "嫌疑人买了一块表"}],
        "fake_events": []
    })
    
    # 存入第一轮供述
    graph.register_testimony("我没买表，我上周都在家。", round_num=1)
    
    # 存入第二轮供述
    graph.register_testimony("表是别人送的。", round_num=2)
    
    engine = EventRetrievalEngine(graph)
    res = engine.retrieve_with_psychology("那块表是怎么回事", {})
    
    context_str = res["kg_context"]
    
    assert "【你自己之前的口供记录 (严禁自相矛盾)】" in context_str
    assert "R1: 我没买表，我上周都在家。" in context_str
    assert "R2: 表是别人送的。" in context_str
    assert "【真实事实 (你必须隐瞒的真相)】" in context_str
