import streamlit as st
import config
from suspect_agent import SuspectAgent
from interrogator import InterrogatorAgent
from api_client import QianwenAPIClient
from referee import Referee
import export_utils
from log_utils import get_logger

logger = get_logger(__name__)

# --- 1. UI 样式  ---
st.set_page_config(page_title="DeepInquisitor - AI 审讯模拟系统", page_icon="🕵️‍♂️", layout="wide")

st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    .sidebar-header { font-size: 1.5rem; font-weight: bold; color: #ff4b4b; margin-bottom: 1rem; }
    .clue-box { padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #eee; }
    .clue-real { background-color: #f0fdf4; border-left: 5px solid #22c55e; }
    .clue-fake { background-color: #fef2f2; border-left: 5px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

try:
    from case_library import CASES
    from personality import PERSONALITY_FACTORY
except:
    CASES = []
    PERSONALITY_FACTORY = {}


# ==========================================
# 核心功能逻辑
# ==========================================

import uuid
from langgraph.checkpoint.memory import MemorySaver

def init_game(personality, rounds, t_ratio):
    api = QianwenAPIClient(config.config.API_BASE_URL, config.config.MODEL_NAME, config.config.API_KEY)
    case = st.session_state.target_case
    interrogator = InterrogatorAgent(api, case['case_data'], truth_ratio=t_ratio)

    intel_logs = []
    for item in interrogator.police_intel_list:
        intel_logs.append({"is_real": item['status'] == "REAL", "p_view": item['content'], "s_view": item['truth']})

    import copy
    current_profile = copy.deepcopy(case['suspect_profile'])
    current_profile['personality'] = personality

    sus_bot = SuspectAgent(
        case['case_data'],
        {"USE_MOCK_LLM": False, "TOTAL_ROUNDS": rounds, "DEFAULT_SUSPECT": current_profile}
    )
    
    referee_bot = Referee(api_client=api, truth_keywords=case.get("keywords",[]))

    # [核心修复] 调用 workflow 构建工厂，获取纯净的图实例
    import workflow
    langgraph_app = workflow.build_langgraph_app(interrogator, sus_bot, referee_bot)
    thread_id = str(uuid.uuid4())

    initial_state = {
        "case_title": case['title'],
        "round_idx": 1,
        "max_rounds": rounds,
        "personality": personality,
        "clue_ratio": t_ratio,
        "history": [],
        "current_question": "审讯正式开始",
        "current_strategy": 3,
        "current_response": "...",
        "defense_value": sus_bot.psych_machine.defense,
        "stress_value": sus_bot.psych_machine.stress,
        "report_data": [],
        "is_broken": False
    }

    st.session_state.game = {
        "lg_app": langgraph_app,
        "thread_config": {"configurable": {"thread_id": thread_id}},
        "agents": {"interrogator": interrogator, "suspect": sus_bot, "referee": referee_bot},
        "intel_logs": intel_logs,
        "started": True,
        "max_rounds": rounds
    }
    
    # 注入初始状态
    langgraph_app.update_state(st.session_state.game["thread_config"], initial_state)


def run_step(u_input=None):
    gs = st.session_state.game
    app = gs["lg_app"]
    t_config = gs["thread_config"]

    if u_input:
        # 人工介入：直接覆盖图状态中的问题节点
        app.update_state(t_config, {"current_question": u_input, "current_strategy": 4}, as_node="interrogator")
        
    # 驱动 LangGraph 执行，直到下一个 interrupt_after (referee)
    app.invoke(None, t_config)

# ==========================================
# UI 布局
# ==========================================

with st.sidebar:
    st.markdown('<div class="sidebar-header">🕵️‍♂️ 审讯配置</div>', unsafe_allow_html=True)
    st.session_state.target_case = st.selectbox("📂 选择审讯案卷", CASES, format_func=lambda x: x['title'])

    st.divider()
    # 动态获取工厂中所有的性格名称
    personality_keys = list(PERSONALITY_FACTORY.keys())
    
    # 默认选中当前关联案件的性格
    default_key = st.session_state.target_case['suspect_profile']['personality']
    default_index = personality_keys.index(default_key) if default_key in personality_keys else 0

    p_type = st.selectbox(
        "🧠 嫌疑人性格设定", 
        personality_keys, 
        index=default_index,
        format_func=lambda k: f"{PERSONALITY_FACTORY[k].name} ({k})" if k in PERSONALITY_FACTORY else k
    )
    max_r = st.slider("⏳ 设定最大交锋轮次", 5, 20, 10)
    t_rate = st.slider("🔍 线索真实度", 0.0, 1.0, 0.5)
    mode = st.radio("🎮 运行模式", ["自动步进", "人工介入提问"])

    if st.button("🚀 开始审讯部署", type="primary", use_container_width=True):
        init_game(p_type, max_r, t_rate)
        st.rerun()

    if st.session_state.get("game"):
        if st.button("🗑️ 终止并重置"):
            st.session_state.clear()
            st.rerun()

# ==========================================
# 主界面渲染
# ==========================================

if st.session_state.get("game") and st.session_state.game["started"]:
    gs = st.session_state.game

    # 战术对照卡片
    with st.expander("📊 内部线索对照（系统视图）", expanded=False):
        for log in gs['intel_logs']:
            c1, c2 = st.columns(2)
            style = "clue-real" if log['is_real'] else "clue-fake"
            with c1: st.markdown(f'<div class="clue-box {style}">👮 警方线索: {log["p_view"]}</div>',
                                 unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="clue-box {style}">🧠 真相记忆: {log["s_view"]}</div>',
                                 unsafe_allow_html=True)

    # 聊天记录 - [LangGraph 重构] 直接从图中拉取当前状态数据
    current_state = gs["lg_app"].get_state(gs["thread_config"]).values
    report_data = current_state.get("report_data", [])
    current_round = current_state.get("round_idx", 1) - 1 # graph round_idx 永远是下一轮
    
    # 模拟重构为兼容导出脚本的数据结构
    gs_compatible = {"messages": [], "current_round": current_round, "max_rounds": gs["max_rounds"], "intel_logs": gs['intel_logs']}
    
    for r in report_data:
        # 重构为旧消息格式兼容 UI 绘制
        gs_compatible["messages"].append({
            "role": "police", "round": r["round"], "content": r["q"]
        })
        gs_compatible["messages"].append({
            "role": "suspect", "round": r["round"], "content": r["a"],
            "psych": {"def": r["def"], "str": r["str"]},
            "ref": {"logic_score": r.get("logic_score", gs_compatible["max_rounds"]), "leak_rate": r.get("leak_rate", 0.0), "conflict": r.get("contradiction", None)}
        })
        
        with st.chat_message("user", avatar="👮"):
            st.write(f"**主审官 第 {r['round']} 轮**")
            st.write(r['q'])
            
        with st.chat_message("assistant", avatar="🦊"):
            st.write(f"**嫌疑人 {st.session_state.target_case['suspect_profile']['name']}**")
            st.write(r['a'])
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("防御值", f"{r['def']:.1f}")
                c2.metric("压力值", f"{r['str']:.1f}")
                c3.metric("逻辑分", int(r.get("logic_score", gs["max_rounds"])))
                c4.metric("泄露率", f"{r.get('leak_rate', 0.0) * 100:.0f}%")
                if r.get("contradiction"): st.error(f"🚩 逻辑矛盾点：{r['contradiction']}")

    # 游戏推进
    curr_def = current_state.get("defense_value", 100)
    curr_logic = report_data[-1].get("logic_score", gs["max_rounds"]) if report_data else gs["max_rounds"]

    if current_round < gs['max_rounds'] and curr_def >= 10 and curr_logic > 0:
        if mode == "人工介入提问":
            if user_q := st.chat_input("输入审讯问题..."): run_step(user_q); st.rerun()
        else:
            with st.spinner("🤖 AI 对抗演算中，正在生成下一轮..."):
                import time
                time.sleep(1) # Optional UI breathing room
                run_step()
            st.rerun()
    else:
        # --- 深度结案总结区 ---
        st.divider()
        st.header("🏁 审讯结案报告")
        final_r = report_data[-1] if report_data else {}
        sus_m = gs["agents"]["suspect"].psych_machine
        
        adjudicate_input = {
            "rounds": report_data,
            "meta": {
                "total_rounds": gs['max_rounds'], 
                "case_title": current_state.get('case_title'),
                "suspect_name": st.session_state.target_case['suspect_profile']['name'],
                "suspect_personality": current_state.get('personality', 'normal'),
                "intel_logs": gs.get('intel_logs', [])
            }
        }
        report = gs['agents']['referee'].adjudicate(adjudicate_input, sus_m)

        with st.container(border=True):
            st.subheader(f"🏆 获胜方：{report.get('winner', '未知')}")
            col1, col2, col3 = st.columns(3)
            col1.metric("最终逻辑分", report.get('logic_score', 0))
            col2.metric("信息泄露率", report.get('leak_rate', '0%'))
            col3.metric("防御削弱总计", f"{100-curr_def:.1f}")

            st.markdown("### 📝 刑侦专家综述")
            st.write(report['summary'])
            
            # 自动导出并显示导出结果
            if "exported_path" not in st.session_state.game:
                st.session_state.game["exported_path"] = export_utils.save_interrogation_report(gs_compatible, report)
            
            if st.session_state.game["exported_path"]:
                st.success(f"💾 本局对抗日志已自动导出至: `{st.session_state.game['exported_path']}`")

            if st.button("🔄 开启新一轮审讯"):
                st.session_state.clear()
                st.rerun()

else:
    # --- 首页 UI 回归 (参考旧版欢迎词) ---
    st.title("🕵️‍♂️ DeepInquisitor 审讯模拟系统")
    st.markdown("""
    ### 欢迎进入沉浸式 AI 审讯模拟环境
    本系统采用多智能体博弈架构，你将扮演主审官，通过心理攻势与证据挖掘破译嫌疑人的心防。

    **核心机制说明：**
    * **心理防御 (Defense)**：通过高压提问降低，低于 10 时嫌疑人将彻底招供。
    * **逻辑自洽 (Logic)**：裁判将实时比对嫌疑人的历次供述。**前后矛盾会导致逻辑分直接扣减。**
    * **信息泄露 (Leakage)**：嫌疑人在回答中提及真实犯罪细节的比例。
    * **战术对照**：系统视角下可展开查看线索的真伪，协助你规划提问路径。

    **👈 请从左侧侧边栏选择案件并点击“开始审讯部署”。**
    """)