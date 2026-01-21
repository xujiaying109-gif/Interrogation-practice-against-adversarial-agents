
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from schemas import AgentState
from graph import build_deep_inquisitor_graph
from interrogator import Interrogator
from judge import Judge


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

def run_autonomous_battle(app, max_turns=10):
    """
    运行 Agent vs Agent 的自动审讯模拟。
    Role 1: 审讯官 (简单 LLM Chain)
    Role 2: 嫌疑人 (DeepInquisitor Graph App)
    """
    print(f"\n{'='*20} ⚔️ DeepInquisitor 自动对战模式 (Max Turns: {max_turns}) ⚔️ {'='*20}\n")
    
    # 1. 初始化审讯官 Agent（独立类，便于后续单独优化）
    interrogator = Interrogator()
    #    初始化裁判系统 (The Judge)
    judge = Judge()
    
    # 2. 初始化对战状态
    chat_history = []  # 用于给审讯官提供上下文
    dialogue_transcript = []  # 完整对话日志，供 Judge 事后评估使用

    # 第一句开场白：可以由审讯官类统一生成，这里先用固定句，然后纳入 history
    initial_query = "张局长，这么晚把你请来，应该知道是为了什么事吧？"
    print(f"👮 [审讯官]: {initial_query}")
    chat_history.append(HumanMessage(content=initial_query)) # 记入历史：Human 是审讯官自己（在 Prompt 逻辑里，为了方便这里反着用也没事，或者用 AI/Human 区分）
    # 更正：在 LangChain Prompt 中，通常 Human 是用户（这里是嫌疑人回复），AI 是模型生成（这里是审讯官问题）。
    # 为了逻辑清晰：
    # history 中：AIMessage = 审讯官的问题，HumanMessage = 嫌疑人的回复。
    # 这样 interrogator_chain 生成的是 AIMessage。
    
    battle_history_for_interrogator = [AIMessage(content=initial_query)]
    current_question = initial_query
    
    # 嫌疑人初始状态
    current_suspect_state = {
        "messages": [HumanMessage(content=initial_query)], # 这里 Human 是审讯官 (对于 Suspect 来说)
        "psych_state": {
            "defense_value": 100.0, 
            "stress_value": 0.0,
            "status_label": "CALM"
        },
        "perception": {},
        "selected_strategy": "",
        "retrieved_knowledge": ""
    }
    
    # 3. 循环对战
    for i in range(max_turns):
        print(f"\n>>> Round {i+1} <<<")
        
        # --- Step A: 嫌疑人 (DeepInquisitor) 行动 ---
        # 调用 Graph App
        try:
            result = app.invoke(current_suspect_state)
            suspect_response = result["messages"][-1].content
            
            # 打印嫌疑人状态和回复
            psych = result["psych_state"]
            print(f"📉 [嫌疑人状态]: 防御 {psych['defense_value']:.1f} | 压力 {psych['stress_value']:.1f} | 模式 {psych['status_label']}")
            print(f"🦊 [张局长]: {suspect_response}")

            # 记录完整对话日志（供 Judge 使用，不做裁剪）
            dialogue_transcript.append(
                {
                    "round": i + 1,
                    "question": current_question,
                    "answer": suspect_response,
                    "psych_state": psych,
                }
            )

            # 记录历史（并裁剪长度，避免上下文无限增长）
            battle_history_for_interrogator.append(HumanMessage(content=suspect_response))
            battle_history_for_interrogator = _trim_history(battle_history_for_interrogator)
            
            # 判断是否结束：
            # 1) 心理防线崩溃（BROKEN）
            # 2) 显式认罪（使用 LLM 语义判断，而非简单关键词匹配）
            # 3) 实质性突破 + 防线显著下滑：承认核心客观事实且解释明显是诡辩（由 Judge 在线粗判），
            #    同时 defense_value 已经跌破一定阈值，避免在完全"强硬防御"状态下被过早判定结束。
            breakthrough = judge.is_substantive_breakthrough_online(suspect_response)
            is_confession = judge.is_confession_online(suspect_response)
            
            if (
                psych["status_label"] == "BROKEN"
                or is_confession
                or (breakthrough and psych["defense_value"] < 40.0)
            ):
                if breakthrough and psych["defense_value"] < 40.0:
                    print(f"\n*** 🎯 审讯成功！嫌疑人在第 {i+1} 轮出现实质性突破（核心事实被承认且解释站不住脚，且心理防线明显动摇）。 ***")
                elif is_confession:
                    print(f"\n*** 🎯 审讯成功！嫌疑人在第 {i+1} 轮明确认罪。 ***")
                else:
                    print(f"\n*** 🎯 审讯成功！嫌疑人在第 {i+1} 轮心理防线崩溃（BROKEN）。 ***")
                break
                
        except Exception as e:
            print(f"❌ 嫌疑人 Agent 运行出错: {e}")
            break
            
        # --- Step B: 审讯官 (Interrogator) 行动 ---
        if i < max_turns - 1: # 最后一轮不需要再问
            try:
                # 生成下一个问题（交给独立的审讯官类）
                next_question = interrogator.ask(battle_history_for_interrogator)
                print(f"👮 [审讯官]: {next_question}")

                # 记录历史（并裁剪长度）
                battle_history_for_interrogator.append(AIMessage(content=next_question))
                battle_history_for_interrogator = _trim_history(battle_history_for_interrogator)

                # 更新嫌疑人状态 (准备下一轮)
                # 关键：必须继承上一轮的 psych_state，否则嫌疑人会“失忆”重置为满血状态
                suspect_messages = result["messages"] + [HumanMessage(content=next_question)]
                suspect_messages = _trim_history(suspect_messages)

                current_suspect_state = {
                    "messages": suspect_messages,
                    "psych_state": result["psych_state"], # <--- 核心：状态传递
                    "perception": {},
                    "selected_strategy": "",
                    "retrieved_knowledge": ""
                }

                # 更新当前审讯问题（用于记录到 transcript 中）
                current_question = next_question
                
            except Exception as e:
                print(f"❌ 审讯官 Agent 运行出错: {e}")
                import traceback
                traceback.print_exc()
                break

    # 4. 对本次审讯过程进行评估（The Judge）
    try:
        judge.evaluate(dialogue_transcript)
    except Exception as e:
        print(f"[Judge Warning] 评估过程出错: {str(e)[:100]}...")

if __name__ == "__main__":
    # 初始化应用
    app = build_deep_inquisitor_graph()
    
    # 运行自动对战
    try:
        run_autonomous_battle(app, max_turns=30)
    except Exception as e:
        print(f"运行时错误 (可能是缺少 API Key): {e}")
