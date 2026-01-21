from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from schemas import AgentState
from graph import build_deep_inquisitor_graph
from interrogator import Interrogator
from judge import Judge

# 导入心理状态机类
from psych_state_machine import DynamicPsychologicalStateMachine


def _trim_history(messages, max_rounds: int = 6):
    """
    简单的上下文裁剪函数：
    - 只保留最近 max_rounds 轮问答（约 2 * max_rounds 条消息）；
    - 这里 messages 中交替包含审讯官和嫌疑人的消息，因此直接按条数裁剪即可。
    """
    if not messages:
        return messages
    max_len = max_rounds * 2
    if len(messages) <= max_len:
        return messages
    return messages[-max_len:]


# ==========================================
# 4. 自动对战模拟 (Autonomous Battle Simulation)
# ==========================================

def run_autonomous_battle(app, max_turns=30):
    """
    运行 Agent vs Agent 的自动审讯模拟。
    Role 1: 审讯官 (简单 LLM Chain)
    Role 2: 嫌疑人 (DeepInquisitor Graph App)
    已集成动态心理状态机
    """
    print(f"\n{'=' * 20} ⚔️ DeepInquisitor 自动对战模式 (Max Turns: {max_turns}) ⚔️ {'=' * 20}\n")

    # 1. 初始化审讯官 Agent（独立类，便于后续单独优化）
    interrogator = Interrogator()
    # 初始化裁判系统 (The Judge)
    judge = Judge()

    # 2. 初始化对战状态
    battle_history_for_interrogator = []  # 用于审讯官的上下文
    dialogue_transcript = []  # 完整对话日志，供 Judge 事后评估使用

    # 第一句开场白
    initial_query = "张局长，这么晚把你请来，应该知道是为了什么事吧？"
    print(f"👮 [审讯官]: {initial_query}")

    # 3. 初始化心理状态机（傲慢型人格）
    psych_machine = DynamicPsychologicalStateMachine("Arrogant")
    initial_psych_state = psych_machine.get_state_dict()

    # 4. 嫌疑人初始状态（集成心理状态机）
    current_suspect_state = {
        "messages": [HumanMessage(content=initial_query)],
        "psych_state": initial_psych_state,
        "psych_machine": psych_machine,  # 添加心理状态机实例
        "profile_type": "Arrogant",  # 添加人格类型
        "perception": {},
        "selected_strategy": "",
        "retrieved_knowledge": ""
    }

    # 记录历史（审讯官角度）
    battle_history_for_interrogator.append(AIMessage(content=initial_query))
    current_question = initial_query

    # 5. 循环对战
    for i in range(max_turns):
        print(f"\n>>> Round {i + 1} <<<")

        # --- Step A: 嫌疑人 (DeepInquisitor) 行动 ---
        # 调用 Graph App
        try:
            result = app.invoke(current_suspect_state)
            suspect_response = result["messages"][-1].content

            # 获取更新后的心理状态
            psych = result.get("psych_state", initial_psych_state)

            # 打印嫌疑人状态和回复
            print(f"📉 [嫌疑人状态]: {psych.get('status_label', 'CALM')} | "
                  f"防御 {psych.get('defense_value', 100.0):.1f} | "
                  f"压力 {psych.get('stress_value', 0.0):.1f}")

            # 如果有风险等级信息，也显示
            if "risk_level" in psych:
                print(f"   ⚠️  崩溃风险: {psych['risk_level']}")

            print(f"🦊 [张局长]: {suspect_response}")

            # 记录完整对话日志（供 Judge 使用）
            dialogue_transcript.append({
                "round": i + 1,
                "question": current_question,
                "answer": suspect_response,
                "psych_state": psych.copy(),  # 深拷贝避免后续修改
                "suspect_profile": "Arrogant"
            })

            # 记录历史（审讯官角度）
            battle_history_for_interrogator.append(HumanMessage(content=suspect_response))
            battle_history_for_interrogator = _trim_history(battle_history_for_interrogator)

            # 判断是否结束条件：
            # 1) 心理防线崩溃（BROKEN 状态）
            # 2) 显式认罪
            # 3) 实质性突破 + 防线显著下滑
            is_broken = result.get("is_broken", False) or psych.get("status_label") == "BROKEN"
            is_confession = judge.is_confession_online(suspect_response)
            breakthrough = judge.is_substantive_breakthrough_online(suspect_response)
            defense_value = psych.get("defense_value", 100.0)

            termination_conditions = [
                (is_broken, "心理防线崩溃（BROKEN状态）"),
                (is_confession, "明确认罪"),
                (breakthrough and defense_value < 40.0, "实质性突破且心理防线明显动摇")
            ]

            should_end = False
            end_reason = ""

            for condition, reason in termination_conditions:
                if condition:
                    should_end = True
                    end_reason = reason
                    break

            if should_end:
                print(f"\n{'=' * 60}")
                print(f"*** 🎯 审讯结束！第 {i + 1} 轮达成终止条件: {end_reason} ***")
                print(f"{'=' * 60}")

                # 显示最终心理状态
                print(f"\n📊 最终心理状态:")
                print(f"   防御值: {defense_value:.1f}/100")
                print(f"   压力值: {psych.get('stress_value', 0.0):.1f}/100")
                print(f"   状态标签: {psych.get('status_label', 'CALM')}")
                print(f"   总轮次: {i + 1}")

                break

        except Exception as e:
            print(f"❌ 嫌疑人 Agent 运行出错: {e}")
            import traceback
            traceback.print_exc()
            break

        # --- Step B: 审讯官 (Interrogator) 行动 ---
        if i < max_turns - 1:  # 最后一轮不需要再问
            try:
                # 生成下一个问题
                next_question = interrogator.ask(battle_history_for_interrogator)
                print(f"👮 [审讯官]: {next_question}")

                # 记录历史
                battle_history_for_interrogator.append(AIMessage(content=next_question))
                battle_history_for_interrogator = _trim_history(battle_history_for_interrogator)

                # 更新嫌疑人状态（关键：保持心理状态连续性）
                # 从上一轮结果获取所有必要状态
                suspect_messages = result.get("messages", current_suspect_state["messages"])
                suspect_messages = suspect_messages + [HumanMessage(content=next_question)]
                suspect_messages = _trim_history(suspect_messages)

                # 获取心理状态机实例（如果存在）
                psych_machine = result.get("psych_machine", current_suspect_state.get("psych_machine"))

                current_suspect_state = {
                    "messages": suspect_messages,
                    "psych_state": result.get("psych_state", initial_psych_state),
                    "psych_machine": psych_machine,  # 传递心理状态机
                    "profile_type": current_suspect_state.get("profile_type", "Arrogant"),
                    "perception": {},
                    "selected_strategy": "",
                    "retrieved_knowledge": ""
                }

                # 更新当前审讯问题
                current_question = next_question

            except Exception as e:
                print(f"❌ 审讯官 Agent 运行出错: {e}")
                import traceback
                traceback.print_exc()
                break

    # 6. 对本次审讯过程进行评估（The Judge）
    print(f"\n{'=' * 60}")
    print("📊 审讯过程评估报告")
    print(f"{'=' * 60}")

    try:
        # 如果有对话记录，进行详细评估
        if dialogue_transcript:
            # 计算一些统计信息
            total_rounds = len(dialogue_transcript)
            final_defense = dialogue_transcript[-1]["psych_state"].get("defense_value", 100.0)
            final_status = dialogue_transcript[-1]["psych_state"].get("status_label", "CALM")
            max_stress = max([t["psych_state"].get("stress_value", 0.0) for t in dialogue_transcript])

            print(f"📈 审讯统计:")
            print(f"   总轮次: {total_rounds}")
            print(f"   最终防御值: {final_defense:.1f}/100")
            print(f"   最终状态: {final_status}")
            print(f"   最大压力值: {max_stress:.1f}/100")

            # 状态变化分析
            status_changes = []
            for t in dialogue_transcript:
                status = t["psych_state"].get("status_label", "CALM")
                if not status_changes or status_changes[-1] != status:
                    status_changes.append(status)

            print(f"   状态变化路径: {' → '.join(status_changes)}")

            # 调用 Judge 进行详细评估
            judge.evaluate(dialogue_transcript)
        else:
            print("⚠️ 无对话记录，无法进行评估")

    except Exception as e:
        print(f"[Judge Warning] 评估过程出错: {str(e)[:100]}...")


def run_interactive_mode(app, max_turns=30):
    """
    交互模式：用户作为审讯官 vs AI嫌疑人
    """
    print(f"\n{'=' * 20} 🎮 DeepInquisitor 交互模式 (Max Turns: {max_turns}) {'=' * 20}\n")
    print("说明: 您将扮演审讯官，AI将扮演嫌疑人张局长")
    print("输入 'quit' 或 '退出' 结束审讯\n")

    # 初始化心理状态机
    psych_machine = DynamicPsychologicalStateMachine("Arrogant")
    initial_psych_state = psych_machine.get_state_dict()

    # 嫌疑人初始状态
    current_suspect_state = {
        "messages": [],
        "psych_state": initial_psych_state,
        "psych_machine": psych_machine,
        "profile_type": "Arrogant",
        "perception": {},
        "selected_strategy": "",
        "retrieved_knowledge": ""
    }

    # 第一轮
    print("👮 请输入您的第一句问话:")
    user_input = input("> ")

    if user_input.lower() in ['quit', '退出', 'exit']:
        print("审讯结束")
        return

    current_suspect_state["messages"] = [HumanMessage(content=user_input)]

    for i in range(max_turns):
        print(f"\n>>> Round {i + 1} <<<")

        try:
            # AI嫌疑人回复
            result = app.invoke(current_suspect_state)
            suspect_response = result["messages"][-1].content
            psych = result.get("psych_state", initial_psych_state)

            # 显示嫌疑人状态
            print(f"\n📉 [嫌疑人状态]: {psych.get('status_label', 'CALM')} | "
                  f"防御 {psych.get('defense_value', 100.0):.1f} | "
                  f"压力 {psych.get('stress_value', 0.0):.1f}")

            print(f"🦊 [张局长]: {suspect_response}")

            # 检查终止条件
            is_broken = result.get("is_broken", False) or psych.get("status_label") == "BROKEN"
            if is_broken:
                print(f"\n*** 🎯 嫌疑人心理防线已崩溃！审讯结束。 ***")
                break

            # 用户继续提问
            print(f"\n👮 请继续提问 (输入 'quit' 退出):")
            user_input = input("> ")

            if user_input.lower() in ['quit', '退出', 'exit']:
                print("审讯结束")
                break

            # 更新状态
            suspect_messages = result.get("messages", current_suspect_state["messages"])
            suspect_messages = suspect_messages + [HumanMessage(content=user_input)]
            suspect_messages = _trim_history(suspect_messages)

            current_suspect_state = {
                "messages": suspect_messages,
                "psych_state": result.get("psych_state", initial_psych_state),
                "psych_machine": result.get("psych_machine", psych_machine),
                "profile_type": current_suspect_state.get("profile_type", "Arrogant"),
                "perception": {},
                "selected_strategy": "",
                "retrieved_knowledge": ""
            }

        except Exception as e:
            print(f"❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()
            break

    # 显示最终状态
    print(f"\n📊 审讯结束 - 最终心理状态:")
    print(f"   防御值: {psych.get('defense_value', 100.0):.1f}/100")
    print(f"   压力值: {psych.get('stress_value', 0.0):.1f}/100")
    print(f"   状态: {psych.get('status_label', 'CALM')}")
    print(f"   总轮次: {i + 1}")


if __name__ == "__main__":
    # 初始化应用
    print("正在初始化 DeepInquisitor 系统...")
    app = build_deep_inquisitor_graph()
    print("系统初始化完成！\n")

    # 选择模式
    print("请选择运行模式:")
    print("1. 自动对战模式 (AI审讯官 vs AI嫌疑人)")
    print("2. 交互模式 (用户审讯官 vs AI嫌疑人)")
    print("3. 退出")

    choice = input("\n请输入选项 (1-3): ").strip()

    try:
        if choice == "1":
            print("\n启动自动对战模式...")
            run_autonomous_battle(app, max_turns=30)
        elif choice == "2":
            print("\n启动交互模式...")
            run_interactive_mode(app, max_turns=30)
        elif choice == "3":
            print("退出程序")
        else:
            print("无效选择，使用默认自动对战模式")
            run_autonomous_battle(app, max_turns=30)
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback

        traceback.print_exc()