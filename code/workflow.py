import os
import json
from datetime import datetime
import traceback
import random
import operator
import re
from typing import Annotated, Dict, List, TypedDict, Union
from langgraph.graph import StateGraph, END

# 导入自定义模块
import config
from case_library import CASES
from suspect_agent import SuspectAgent
from interrogator import InterrogatorAgent
from api_client import QianwenAPIClient
from referee import Referee
from color_utils import UI, Colors
from psych_state import MentalStateMachine
from event_kg_module import EventRetrievalEngine, EventGraph

# 1. 状态定义
class AgentState(TypedDict):
    case_title: str
    round_idx: int
    max_rounds: int
    personality: str
    clue_ratio: float
    history: Annotated[List[Dict], operator.add]
    current_question: str
    current_strategy: int
    current_response: str
    defense_value: float
    stress_value: float
    is_broken: bool
    report_data: Annotated[List[Dict], operator.add]
    last_conflict: str
    current_turn_data: Dict


# 3. 构建图
from langgraph.checkpoint.memory import MemorySaver

def build_langgraph_app(interrogator_agent, suspect_agent, referee_agent):
    """
    工厂函数：构建并返回绑定了特定智能体实例的 LangGraph 编译应用。
    使用闭包避免多租户/多会话下的全局变量污染。
    """
    # 2. 节点逻辑 (作为内部函数以访问外部传入的 agent 实例)
    def should_continue(state: AgentState):
        if state['is_broken'] or state['round_idx'] > state['max_rounds']:
            return "end"
        return "continue"

    def interrogator_node(state: AgentState):
        UI.print_role_header(f"第 {state['round_idx']} / {state['max_rounds']} 轮", "⚖️", Colors.SYSTEM)
        print(f"{Colors.DIM}[审讯官正在分析历史记录并挑选情报线索...]{Colors.RESET}")
    
        # 调用审讯官：返回 (分析, 策略ID, 问题文本, 曝光证据ID)
        res = interrogator_agent.generate_question(state['current_response'])
    
        if isinstance(res, tuple) and len(res) >= 4:
            analysis, strat_id, question_text, focus_ev_id = res
            print(f"{Colors.DIM}💭 {analysis}{Colors.RESET}\n")
            # 记录此证据已被当面抛出，嫌疑人无法继续装傻
            suspect_agent.event_graph.mark_evidence_exposed(focus_ev_id)
        elif isinstance(res, tuple) and len(res) >= 3:
            analysis, strat_id, question_text = res
            print(f"{Colors.DIM}💭 {analysis}{Colors.RESET}\n")
        else:
            strat_id = 3
            question_text = str(res)
    
        UI.print_dialogue(Colors.POLICE, "审讯官", question_text)
        s_map = {1: "心理施压", 2: "出示证据", 3: "连续追问", 4: "正常询问", 5: "缓和安抚"}
        strat_name = s_map.get(strat_id, "正常询问")
    
        # 存入一个临时变量，以便 suspect_node 封装数据时使用
        return {"current_question": question_text, "current_strategy": strat_id, "last_strat_name": strat_name}
    
    
    def suspect_node(state: AgentState):
        """嫌疑人心理博弈节点"""
        res = suspect_agent.process_interrogation(
            state['current_question'],
            state['current_strategy']
        )
    
        response_text = res[0] if isinstance(res, tuple) else str(res)
        psych = suspect_agent.psych_machine
        sus_name = state['case_title'].split('案')[0] + "某"
    
        UI.print_dialogue(Colors.SUSPECT, sus_name, response_text)
    
        # 封装当轮数据给裁判
        current_turn_data = {
            "q": state['current_question'],
            "a": response_text,
            "def": psych.defense,
            "str": psych.stress,
            "round": state['round_idx'],
            "police_strat": state.get("last_strat_name", "询问")  # 获取策略名
        }
    
        return {
            "current_response": response_text,
            "history": [{"role": "police", "content": state['current_question']},
                        {"role": "suspect", "content": response_text}],
            "defense_value": psych.defense,
            "stress_value": psych.stress,
            "current_turn_data": current_turn_data,
            "is_broken": psych.defense < 10
        }
    
    
    def referee_node(state: AgentState):
        """功能2实现：裁判方介入，仅比较历史口供"""
        current_turn_data = state.get('current_turn_data', {})
        
        # 1. 准备历史口供（不含本轮警方的线索，只含嫌疑人的回答）
        history_rounds = state['report_data']
    
        # 2. 调用实时逻辑校验
        conflict_res = referee_agent.verify_realtime_logic(
            state['current_response'],
            history_rounds
        )
    
        # 3. 如果检测到矛盾，进行高亮打印
        if conflict_res and conflict_res.get("conflict"):
            print(f"\n{Colors.REFEREE}⚠️  [裁判介入] 检测到矛盾{Colors.RESET}")
            print(f"{Colors.REFEREE}   └─ {conflict_res['conflict']}{Colors.RESET}")
            # 存入本轮报告
            current_turn_data["contradiction"] = conflict_res["conflict"]
            
        # --- 计算全链路的累积逻辑分和泄露率，对接 Streamlit UI ---
        past_conflicts = sum(1 for r in state['report_data'] if r.get("contradiction"))
        if current_turn_data.get("contradiction"):
            past_conflicts += 1
            
        cumulative_logic_score = float(state['max_rounds']) - past_conflicts
        
        # 将历史回答 + 本次最新回答拼起来，送给大模型进行语义判重
        all_responses = " ".join([r.get('a', '') for r in state['report_data']] + [current_turn_data.get('a', '')])
        matched = referee_agent.semantic_leak_check(all_responses)
        # 计算总证据点数量作为分母
        total_evidence_count = 0
        if referee_agent.case_facts:
            for fact in referee_agent.case_facts:
                total_evidence_count += len(fact.get("evidence_points", []))
        leak_rate = len(matched) / total_evidence_count if total_evidence_count > 0 else 0.0
        
        current_turn_data["logic_score"] = cumulative_logic_score
        current_turn_data["leak_rate"] = leak_rate
    
        print(f"\n📊 状态监控 | 防御: {state['defense_value']:.1f} | 压力: {state['stress_value']:.1f}")
        return {
            "last_conflict": conflict_res.get("conflict", "") if conflict_res else "",
            "report_data": [current_turn_data],
            "round_idx": state['round_idx'] + 1
        }
    
    workflow = StateGraph(AgentState)
    workflow.add_node("interrogator", interrogator_node)
    workflow.add_node("suspect", suspect_node)
    workflow.add_node("referee", referee_node)
    
    workflow.set_entry_point("interrogator")
    workflow.add_edge("interrogator", "suspect")
    workflow.add_edge("suspect", "referee")
    workflow.add_conditional_edges("referee", should_continue, {"continue": "interrogator", "end": END})
    
    # 增加中断点，允许前端分发渲染
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, interrupt_after=["referee"])
    return app

# --- 优化后的功能1实现：认知差构建展示函数 ---
def display_intelligence_matrix(interrogator, case_data):
    """解析审讯官的情报网并对照案件库真相展示"""
    # 强制刷新缓冲区，确保标题先出来
    print("\n" + "=" * 60)
    UI.print_role_header("💉 认知差构建 (记忆注入)", "对比警方掌握的情报与实际发生的真相", Colors.SYSTEM)
    print("=" * 60)

    for item in interrogator.police_intel_list:
        content = item['content']
        real_truth = item['truth']
        if item['status'] == "FAKE":
            # 虚假情报
            tag = "❌ 虚假情报 (警方的错误/伪造信息)"
            header_color = Colors.REFEREE
            truth_label = "🧠 嫌疑人真实记忆"
        else:
            # 真实情报
            tag = "✅ 真实线索 (警方掌握的事实)"
            header_color = Colors.SYSTEM
            truth_label = "🧠 嫌疑人真实记忆"

        print(f"\n{header_color}{tag}{Colors.RESET}")
        print(f"  {Colors.POLICE}警方视角:{Colors.RESET} {content}")
        print(f"  {Colors.SUSPECT}{truth_label}:{Colors.RESET} {real_truth}")
        print("-" * 40)

    print("\n" + "=" * 60)


def save_interrogation_report(final_output: Dict, referee_report: str, interrogator_obj):
    """
    保存审讯全过程报告，修复了 Colors.UNDERLINE 报错问题
    """
    save_dir = "saved_reports"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 清理文件名中的非法字符
    case_title = final_output.get("case_title", "unknown_case").replace(" ", "_")
    filename = f"{save_dir}/{timestamp}_{case_title}.md"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# 🕵️ DeepInquisitor 审讯报告 (LangGraph版)\n")
            f.write(f"- **案件**: {case_title}\n")
            f.write(f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **最终防御值**: {final_output.get('defense_value', 0):.1f}\n")
            f.write(f"- **结果**: {'突破成功' if final_output.get('is_broken') else '未能突破'}\n\n")

            # 1. 裁判评估
            f.write("## 1. 裁判评估\n```text\n" + referee_report + "\n```\n\n")

            # 2. 认知差构建记录
            f.write("## 2. 认知差构建 (记忆注入)\n")
            if hasattr(interrogator_obj, 'police_intel_list'):
                for item in interrogator_obj.police_intel_list:
                    status = "真实" if item['status'] == "REAL" else "虚假"
                    f.write(f"### [{status}] 线索 {item['id']}\n")
                    f.write(f"- **警方掌握**: {item['content']}\n")
                    f.write(f"- **实际真相**: {item['truth']}\n\n")

            # 3. 对话全记录
            f.write("## 3. 详细审讯记录\n")
            for r in final_output.get('report_data', []):
                f.write(f"### 第 {r['round']} 轮\n")
                # 兼容处理策略名称
                strat = r.get('police_strat', '正常询问')
                f.write(f"**审讯官 ({strat})**: {r['q']}\n\n")
                f.write(f"**嫌疑人**: {r['a']}\n\n")
                f.write(f"**状态监控**: 防御 {r['def']:.1f} | 压力 {r['str']:.1f}\n")
                if r.get("contradiction"):
                    f.write(f"\n> ⚠️ **裁判介入 (检测到矛盾)**: {r['contradiction']}\n")
                f.write("\n---\n")

        # 修复点：移除了 Colors.UNDERLINE
        print(f"\n{Colors.SYSTEM}✅ 审讯记录已保存至: {filename}{Colors.RESET}")

    except Exception as e:
        print(f"\n{Colors.REFEREE}❌ 保存报告失败: {str(e)}{Colors.RESET}")
        traceback.print_exc()

# 4. 运行入口
if __name__ == "__main__":
    selected_case = random.choice(CASES)
    UI.print_role_header("DeepInquisitor 启动中...", "🚀", Colors.SYSTEM)
    print(f"📂 案件: {selected_case['title']}")
    print(f"👤 嫌疑人: {selected_case['suspect_profile']['name']}")

    # 配置
    print("请选择嫌疑人性格 (1. 紧张焦虑 2. 傲慢自大 3. 冷静谨慎): ", end="")
    p_choice = input()
    p_map = {"1": "nervous", "2": "arrogant", "3": "cautious"}
    personality_str = p_map.get(p_choice, "arrogant")
    rounds = int(input("轮次 [5]: ") or 5)
    ratio = float(input("真实线索比例 [0.5]: ") or 0.5)

    # 初始化组件
    agent_config = config.AgentConfig()
    api_client = QianwenAPIClient(agent_config.API_BASE_URL, agent_config.MODEL_NAME, agent_config.API_KEY)
    # 初始化各智能体
    interrogator_agent = InterrogatorAgent(api_client, selected_case['case_data'], truth_ratio=ratio)
    import copy
    current_profile = copy.deepcopy(selected_case['suspect_profile'])
    current_profile['personality'] = personality_str

    suspect_agent = SuspectAgent(
        selected_case['case_data'],
        {"USE_MOCK_LLM": False, "DEFAULT_SUSPECT": current_profile, "TOTAL_ROUNDS": rounds}
    )
    suspect_agent.psych_machine.total_rounds = rounds
    referee_agent = Referee(api_client=api_client, case_facts=selected_case['case_data'].get("facts", []))

    # --- 执行功能1展示 ---
    display_intelligence_matrix(interrogator_agent, selected_case["case_data"])

    initial_state = {
        "case_title": selected_case['title'],
        "round_idx": 1,
        "max_rounds": rounds,
        "personality": personality_str,
        "clue_ratio": ratio,
        "history": [],
        "current_question": "审讯正式开始",
        "current_strategy": 3,
        "current_response": "...",
        "defense_value": suspect_agent.psych_machine.defense,
        "stress_value": suspect_agent.psych_machine.stress,
        "report_data": [],
        "is_broken": False
    }

    input("\n情报注入完成，按回车键开始审讯循环 >>")
    # 启动图
    final_output = app.invoke(initial_state)

    # --- 最终评估报告 ---
    UI.print_role_header("结案报告", "📋", Colors.SYSTEM)
    adjudicate_input = {
        "rounds": final_output['report_data'],
        "meta": {"total_rounds": rounds, "case_title": final_output['case_title']}
    }
    
    report = referee_agent.adjudicate(adjudicate_input, suspect_agent.psych_machine)
    print(report)
    print(referee_agent.adjudicate(adjudicate_input, suspect_agent.psych_machine))
    # --- [新增] 保存全过程记录 ---
    save_interrogation_report(final_output, report, interrogator_agent)