# main_integrated.py
from langchain_core.messages import HumanMessage
from schemas import AgentState
from graph import build_deep_inquisitor_graph
from psych_state_machine import DynamicPsychologicalStateMachine
from interrogator_memory import InterrogatorMemory  # 导入完整的记忆系统
import datetime


def run_interrogation_with_memory(app, max_turns=30):
    """
    完整集成记忆系统的审讯程序
    """
    print(f"\n{'=' * 60}")
    print("🔍 DeepInquisitor 审讯系统（集成记忆版）")
    print(f"{'=' * 60}\n")

    print("🎮 审讯模式：用户扮演审讯官 vs AI嫌疑人")
    print("💾 系统特性：自动记忆、矛盾检测、智能建议")
    print("📋 输入 'help' 查看所有可用命令\n")

    # 初始化嫌疑人心理状态
    print("正在初始化嫌疑人心理状态...")
    psych_machine = DynamicPsychologicalStateMachine("Arrogant")
    initial_psych_state = psych_machine.get_state_dict()

    # 初始化审讯官记忆系统（核心记忆存储）
    print("正在初始化审讯官记忆系统...")
    interrogator_memory = InterrogatorMemory("张局长")

    # 初始化嫌疑人状态
    current_suspect_state = {
        "messages": [],
        "psych_state": initial_psych_state,
        "psych_machine": psych_machine,
        "profile_type": "Arrogant",
        "perception": {},
        "selected_strategy": "",
        "retrieved_knowledge": ""
    }

    # 显示审讯基本信息
    print(f"嫌疑人：张局长（傲慢型人格）")
    print(f"最大审讯轮次：{max_turns}")
    print(f"初始防御值：{initial_psych_state['defense_value']}")
    print(f"初始压力值：{initial_psych_state['stress_value']}")
    print(f"\n{'=' * 60}\n")

    # 开始审讯
    print("👮 请输入第一句问话（或输入 'help' 获取帮助）：")
    user_input = input("> ").strip()

    # 处理初始命令
    if user_input.lower() in ['quit', '退出', 'exit']:
        print("审讯结束")
        return

    if user_input.lower() == 'help':
        show_help()
        print("\n请继续提问：")
        user_input = input("> ").strip()

    # 第一轮审讯
    current_suspect_state["messages"] = [HumanMessage(content=user_input)]

    # 主审讯循环
    for i in range(max_turns):
        print(f"\n{'=' * 60}")
        print(f"📊 第 {i + 1} 轮审讯")
        print(f"{'=' * 60}")

        # 更新审讯阶段
        interrogator_memory.update_interrogation_phase(i + 1, max_turns)

        try:
            # ============ 嫌疑人回答 ============
            print("\n🤖 AI嫌疑人正在思考...")
            result = app.invoke(current_suspect_state)
            suspect_response = result["messages"][-1].content
            psych = result.get("psych_state", initial_psych_state)

            # 显示嫌疑人状态
            print(f"\n🎭 嫌疑人状态：")
            print(f"  心理状态：{psych.get('status_label', 'CALM')}")
            print(f"  防御值：{psych.get('defense_value', 100.0):.1f}/100")
            print(f"  压力值：{psych.get('stress_value', 0.0):.1f}/100")

            if "risk_level" in psych:
                print(f"  崩溃风险：{psych['risk_level']}")

            # 显示回答
            print(f"\n🦊 [张局长]：{suspect_response}")

            # ============ 核心记忆存储 ============
            print(f"\n💾 正在存储到记忆系统...")

            # 创建元数据
            metadata = {
                "defense_value": psych.get('defense_value', 100.0),
                "stress_value": psych.get('stress_value', 0.0),
                "status_label": psych.get('status_label', 'CALM'),
                "risk_level": psych.get('risk_level', '未知'),
                "strategy": result.get("selected_strategy", "未知")
            }

            # 存储本轮对话到核心记忆
            interrogator_memory.add_conversation(user_input, suspect_response, metadata)

            print(f"✅ 第{i + 1}轮对话已存储到记忆")

            # ============ 检查终止条件 ============
            is_broken = result.get("is_broken", False) or psych.get("status_label") == "BROKEN"
            if is_broken:
                print(f"\n{'=' * 60}")
                print("🎯 嫌疑人心理防线已崩溃！审讯结束。")
                print(f"{'=' * 60}")
                break

            # ============ 每3轮显示记忆摘要 ============
            if (i + 1) % 3 == 0:
                print(f"\n📋 [记忆系统摘要 - 第{i + 1}轮]")
                print("-" * 50)
                print(interrogator_memory.get_memory_summary())
                print("-" * 50)

            # ============ 获取用户下一轮输入 ============
            while True:
                print(f"\n👮 请输入第{i + 2}轮问话或命令：")
                next_input = input("> ").strip()

                # 处理命令
                if next_input.lower() in ['quit', '退出', 'exit']:
                    print("审讯结束")
                    return

                elif next_input.lower() == 'help':
                    show_help()
                    continue

                elif next_input.lower() == 'summary':
                    print(f"\n📋 [详细记忆摘要]")
                    print("=" * 60)
                    print(interrogator_memory.get_memory_summary())
                    print("=" * 60)
                    continue

                elif next_input.lower() == 'suggest':
                    suggestions = interrogator_memory.suggest_next_questions(3)
                    print(f"\n💡 [问题建议]")
                    for idx, suggestion in enumerate(suggestions, 1):
                        print(f"  {idx}. {suggestion}")
                    print("\n您可以选择使用建议的问题，或提出自己的问题")
                    continue

                elif next_input.lower() == 'history':
                    context = interrogator_memory.get_conversation_context(5)
                    print(f"\n📜 [对话历史]")
                    print(context)
                    continue

                elif next_input.lower() == 'contradictions':
                    contradictions = interrogator_memory.contradictions
                    if contradictions:
                        print(f"\n⚠️  [发现的矛盾点]")
                        for c in contradictions[-5:]:
                            print(f"  - 第{c['current_round']}轮 vs 第{c['previous_round']}轮")
                            print(f"    描述：{c['description']}")
                            print(f"    严重程度：{c.get('severity', '未知')}")
                            print()
                    else:
                        print("暂无发现的矛盾点")
                    continue

                elif next_input.lower() == 'facts':
                    facts = interrogator_memory.confirmed_facts
                    if facts:
                        print(f"\n✅ [已确认的事实]")
                        for f in facts[-5:]:
                            print(f"  - 第{f['round']}轮：{f['fact']}")
                            print(f"    置信度：{f.get('confidence', 0.5):.1f}")
                            print()
                    else:
                        print("暂无已确认的事实")
                    continue

                elif next_input.lower().startswith('save '):
                    filename = next_input[5:].strip()
                    if not filename:
                        filename = f"interrogation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    if not filename.endswith('.json'):
                        filename += '.json'
                    interrogator_memory.save_to_file(filename)
                    continue

                elif next_input.lower().startswith('load '):
                    filename = next_input[5:].strip()
                    if interrogator_memory.load_from_file(filename):
                        print("记忆加载成功！")
                        # 更新当前轮次
                        i = len(interrogator_memory.conversation_history) - 1
                    continue

                elif next_input.lower() == 'profile':
                    profile = interrogator_memory.suspect_profile
                    print(f"\n👤 [嫌疑人特征档案]")
                    for category, items in profile.items():
                        if items:
                            print(f"  {category}：")
                            for item in items[:3]:  # 显示最近3条
                                print(f"    - 第{item['round']}轮：{item.get('context', '')[:60]}...")
                    continue

                elif next_input.lower() == 'stats':
                    stats = interrogator_memory.stats
                    print(f"\n📈 [审讯统计]")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                elif next_input.lower() == 'phase':
                    print(f"\n🎯 当前审讯阶段：{interrogator_memory.interrogation_phase}")
                    print(f"   施压级别：{interrogator_memory.pressure_level}/10")
                    continue

                # ============ 正常问题输入 ============
                user_input = next_input

                # 检查是否重复提问（核心记忆检查）
                normalized_question = interrogator_memory._normalize_text(user_input)
                if normalized_question in interrogator_memory.asked_questions:
                    print(f"⚠️  注意：这个问题或类似问题之前已经问过了！")
                    print(f"    相似问题记录：")
                    for idx, q in enumerate(interrogator_memory.asked_questions_raw[-3:], 1):
                        print(f"    {idx}. {q[:60]}...")
                    print(f"\n确定要继续问这个问题吗？(y/n)")
                    confirm = input("> ").lower()
                    if confirm != 'y':
                        continue

                break

            # ============ 更新嫌疑人状态 ============
            suspect_messages = result.get("messages", current_suspect_state["messages"])
            suspect_messages = suspect_messages + [HumanMessage(content=user_input)]

            # 保持消息历史长度
            if len(suspect_messages) > 10:
                suspect_messages = suspect_messages[-10:]

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

    # ============ 审讯结束 ============
    print(f"\n{'=' * 60}")
    print("📊 审讯结束 - 最终报告")
    print(f"{'=' * 60}")

    print(f"总轮次：{i + 1}")
    print(f"最终心理状态：{psych.get('status_label', 'CALM')}")
    print(f"最终防御值：{psych.get('defense_value', 100.0):.1f}/100")
    print(f"最终压力值：{psych.get('stress_value', 0.0):.1f}/100")

    # 记忆系统统计
    print(f"\n💾 记忆系统统计：")
    print(f"  存储对话：{len(interrogator_memory.conversation_history)} 轮")
    print(f"  确认事实：{len(interrogator_memory.confirmed_facts)} 条")
    print(f"  发现矛盾：{len(interrogator_memory.contradictions)} 处")
    print(f"  嫌疑人特征：{sum(len(v) for v in interrogator_memory.suspect_profile.values())} 条")

    # 询问是否保存
    print(f"\n💾 是否保存完整的审讯记录？(y/n)")
    if input("> ").lower() == 'y':
        default_name = f"interrogation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"请输入文件名（默认：{default_name}）：")
        filename = input("> ").strip()
        if not filename:
            filename = default_name
        interrogator_memory.save_to_file(filename)


def show_help():
    """显示帮助信息"""
    print("\n📋 可用命令：")
    print("  summary        - 查看记忆摘要")
    print("  suggest        - 获取问题建议")
    print("  history        - 查看对话历史")
    print("  contradictions - 查看发现的矛盾点")
    print("  facts          - 查看已确认的事实")
    print("  profile        - 查看嫌疑人特征档案")
    print("  stats          - 查看审讯统计")
    print("  phase          - 查看当前审讯阶段")
    print("  save <文件名>   - 保存审讯记录")
    print("  load <文件名>   - 加载审讯记录")
    print("  help           - 显示此帮助")
    print("  quit           - 退出审讯")
    print("\n💡 提示：直接输入问题开始审讯")


if __name__ == "__main__":
    print("正在初始化 DeepInquisitor 系统...")
    app = build_deep_inquisitor_graph()
    print("系统初始化完成！\n")

    try:
        run_interrogation_with_memory(app, max_turns=30)
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback

        traceback.print_exc()