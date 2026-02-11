# case_data.py
from schemas import DeceptionOperator

CASES = {
    "case_001": {
        "profile": {
            "name": "张建设",
            "position": "财政局副局长",
            "personality": "老练、多疑",
            "background": "在位 10 年"
        },
        "ground_truth": [
            {
                "subject": "张建设", 
                "predicate": "收受", 
                "object": "50万现金", 
                "evidence_strength": 0.8, 
                "meta": {"date": "2023年1月5日", "location": "云隐茶馆"}, 
                "narrative": "张建设在茶馆收受了贿赂。"
            },
            {
                "subject": "50万现金", 
                "predicate": "存放于", 
                "object": "办公室保险柜", 
                "evidence_strength": 0.3, 
                "meta": {"status": "赃款未动"}, 
                "narrative": "赃款藏在保险柜里。"
            }
        ],
        "fake_story": [
            {
                "subject": "50万现金", 
                "predicate": "属于", 
                "object": "合法借款", 
                "evidence_strength": 0.7, 
                "meta": {"operator": DeceptionOperator.DISTORT}, 
                "narrative": "那50万是我找李老板借的周转资金，我有借条，那是合法借款。"
            },
            {
                "subject": "张建设", 
                "predicate": "调解", 
                "object": "邻里纠纷", 
                "evidence_strength": 0.6,
                "meta": {"operator": DeceptionOperator.RATIONALIZE}, 
                "narrative": "那天去茶馆是为了帮老部下调解邻里矛盾，属于私人帮忙，不违规。"
            }
        ],
        "context_memory": []
    },
    
    "case_002": {
        "profile": {
            "name": "王秘书",
            "position": "机要办职员"
        },
        "ground_truth": [
            {
                "subject": "王秘书", 
                "predicate": "拍摄", 
                "object": "保密文件", 
                "evidence_strength": 0.5, 
                "meta": {"method": "手机拍照"}, 
                "narrative": "私自拍摄保密文件。"
            },
            {
                "subject": "王秘书", 
                "predicate": "传送", 
                "object": "境外中介Q", 
                "evidence_strength": 0.9, 
                "meta": {"platform": "Telegram"}, 
                "narrative": "发送泄密资料。"
            }
        ],
        "fake_story": [
            {
                "subject": "王秘书", 
                "predicate": "修理", 
                "object": "碎纸机", 
                "evidence_strength": 0.4,
                "meta": {"operator": DeceptionOperator.FABRICATE}, 
                "narrative": "那天碎纸机坏了，我弯腰在那是为了修机器，根本没碰过什么文件。"
            }
        ],
        "context_memory": []
    }
}