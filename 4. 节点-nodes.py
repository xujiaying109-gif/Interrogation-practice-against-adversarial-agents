
import json
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm
from schemas import AgentState, PerceptionOutput
from character import CHARACTER_PROFILE
from knowledge import GLOBAL_KG
from utils import parse_json_from_llm

# ==========================================
# 2. 核心逻辑节点 (Nodes)
# ==========================================

# --- 节点 A: 感知 (Perception) ---
def perception_node(state: AgentState):
    """
    对应论文: 4.3 Step 1 Perception (Intent Analysis)
    使用 LLM 分析用户输入的意图、证据强度和是否包含陷阱。
    """
    last_message = state["messages"][-1].content
    print(f"\n--- [1. 感知阶段] 正在分析输入: '{last_message}' ---")
    
    llm = get_llm(temperature=0.0)
    
    # [Context Retrieval]
    context_info = GLOBAL_KG.retrieve_all_context(last_message)
    print(f"   -> [Perception Context]: {context_info[:100]}...")

    system_prompt = """你是一个嫌疑人AI的'内部感知模块'。你的任务是分析审讯官的每一句话。
    
    **Knowledge Context:**
    {context}
    
    **分析原则 (CRITICAL):**
    1. **Strict Context Checking**: Only claim input matches 'Ground Truth' or 'Fake Story' if the {context} explicitly contains that information.
    2. If {context} is empty or says "无相关背景信息", do NOT hallucinate that the input matches anything. Treat it as "UNKNOWN" or "NEW CLAIM".
    
    1. **High Threat**: Input matches info in 【客观事实 (Ground Truth)】 but contradicts or is missing from 【当前供词 (Fake Story)】.
    2. **Consistency Check**: Input matches info in 【当前供词 (Fake Story)】.
    3. **Low Threat**: Input matches info in 【背景记忆 (Context)】.
    4. **Trap Detection**: If input claims to have "evidence" (e.g., video, logs) that is NOT in your Ground Truth, it might be a bluff/trap. But if it MATCHES Ground Truth, it is a deadly trap.
    
    请输出 JSON 对象，字段如下：
    {{
        "intent": "EVIDENCE_PRESENTATION",
        "evidence_strength": 0.0-1.0,
        "is_trap": true/false,
        "analysis": "..."
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "input": last_message,
            "context": context_info
        })
        content = response.content.strip()
        
        parsed_data = parse_json_from_llm(content)
        perception_result = PerceptionOutput(**parsed_data)
        
        print(f"   -> [LLM Analysis]: {perception_result.analysis}")
        print(f"   -> [Data]: Intent={perception_result.intent}, Trap={perception_result.is_trap}, Strength={perception_result.evidence_strength}")
        
    except Exception as e:
        print(f"   -> [System Warning] JSON Parsing Failed: {str(e)[:100]}...")
        perception_result = PerceptionOutput(
            intent="PRESSURE",
            evidence_strength=0.5,
            is_trap=False,
            analysis=f"[System Fallback] Model output format error. Treated as generic pressure."
        )
    
    return {"perception": perception_result.model_dump()}

# --- 节点 B: 心理状态机 (Linear Dynamical System Update) ---
def psych_update_node(state: AgentState):
    """
    对应论文: 4.2 Dynamic Psychological State Machine (DPSM)
    Uses a Linear Dynamical System (LDS): S(t+1) = A*S(t) + B*U(t)
    
    State Vector S(t): [Defense, Stress]^T
    Input Vector U(t): [Evidence_Strength, Trap_Impact]^T
    
    Trap_Impact: +1 (Trap Detected), 0 (Neutral), -1 (Trap Triggered/Unknown)
    """
    print("--- [2. 心理演变] 更新状态机 (LDS Model) ---")
    
    current_psych = state["psych_state"]
    perception = state["perception"]
    profile = CHARACTER_PROFILE["psychometrics"]
    
    # 1. State Vector S_t
    D_t = current_psych["defense_value"]
    S_t = current_psych["stress_value"]
    
    # 2. Input Vector U_t
    E_t = perception["evidence_strength"]
    
    # Trap Impact Logic
    # Trap Detected (+1): Boost Defense, Reduce Stress
    # Trap Triggered (-1): Reduce Defense, Boost Stress (If trap exists but not detected, effectively triggered)
    # Neutral (0): No trap involved
    T_t = 0.0
    if perception["is_trap"]:
        T_t = 1.0 # Detected!
    # Note: If is_trap=False but intent=PRESSURE, we might consider it neutral or negative. 
    # For now, simplistic mapping: Trap detected = +1 gain.
    defense_current = current_psych["defense_value"]
    stress_current = current_psych["stress_value"]
    
    # --- Non-Linear DPSM Logic ---
    
    # 1. Inputs
    E_strength = perception["evidence_strength"]
    # Pressure is high if Trap is True (mental load)
    Pressure = 1.0 if perception["is_trap"] else 0.0
    
    # 2. Stress Update: S_{t+1} = gamma * S_t + alpha * E + beta * P
    # gamma: Natural decay (1 - resilience/2)
    gamma_decay = 1.0 - (0.25 * profile["resilience"])
    alpha_stress_gain = 18.0 * (1.0 + profile["volatility"])  # Evidence adds stress (减弱整体应激幅度)
    beta_trap_stress = 6.0  # Traps add pressure even if spotted (略微降低)
    
    stress_new = (gamma_decay * stress_current) + (alpha_stress_gain * E_strength) + (beta_trap_stress * Pressure)
    stress_new = max(0.0, min(100.0, stress_new))
    
    # 3. Defense Update: D_{t+1} = D_t - Damage + Recovery
    # Damage Function D(.): Amplified by Stress
    # 仅当证据强度中等偏上时才产生实质性“破防”效果，弱证据主要通过压力慢慢侵蚀。
    # Damage = Base * max(E - threshold, 0) * (1 + Stress/120)
    base_damage = 14.0  # 略微降低基础伤害
    threshold = 0.35    # 低于该强度时认为对防线的直接打击有限
    effective_E = max(0.0, E_strength - threshold)
    damage = base_damage * effective_E * (1.0 + (stress_current / 120.0))
    
    # Recovery Function R(.): Boost if Trap detected
    # Success_trap: True only if it WAS a trap AND we spotted it (Trap=True) 
    # Current perception logic: perception.is_trap means "I think it is a trap". 
    # If I think it's a trap, I get a boost.
    recovery = 0.0
    if perception["is_trap"]:
        recovery = 10.0 + (5.0 * profile["resistance"])
        
    defense_new = defense_current - damage + recovery
    defense_new = max(0.0, min(100.0, defense_new))
    
    print(f"   -> [DPSM Trace] D: {defense_current:.1f}->{defense_new:.1f} (-{damage:.1f}/+{recovery:.1f}) | S: {stress_current:.1f}->{stress_new:.1f}")
    
    # 6. Apply Constraints (already done in new logic)
    
    # 7. Discrete Labeling
    if defense_new < 20: label = "BROKEN"
    elif defense_new < 50: label = "ANXIOUS"
    elif stress_new > 70: label = "DEFENSIVE"
    else: label = "CALM"
    
    updated_psych = {
        "defense_value": defense_new,
        "stress_value": stress_new,
        "status_label": label
    }
    
    print(f"   -> 更新后状态: {updated_psych}")
    
    return {"psych_state": updated_psych}

# --- 节点 C: 策略选择 (Strategy Selection) ---
def strategy_node(state: AgentState):
    """
    对应论文: 4.3 Step 2 Strategic Decision
    """
    print("--- [3. 策略决策] 选择对抗手段 ---")
    psych = state["psych_state"]
    perception = state["perception"]
    status = psych["status_label"]
    evidence_strength = perception["evidence_strength"]
    defense_value = psych["defense_value"]
    
    # 当证据很强但防线尚未完全崩溃时，嫌疑人更理性地选择“部分让步”，
    # 主动承认一些相对次要/不致命的事实，以换取整体说法的可信度。
    if status != "BROKEN" and evidence_strength > 0.7 and defense_value > 40.0:
        strategy = "PARTIAL_CONCESSION"
    elif status == "BROKEN":
        strategy = "FULL_CONFESSION"
    elif status == "CALM" and perception["is_trap"]:
        strategy = "FEIGN_IGNORANCE"
    elif status == "DEFENSIVE" and evidence_strength > 0.5:
        strategy = "RED_HERRING"
    elif status == "ANXIOUS":
        strategy = "INFORMATION_DILUTION"
    else:
        strategy = "DIRECT_DENIAL"
        
    print(f"   -> 选定策略: [{strategy}]")
    return {"selected_strategy": strategy}

# --- 节点 D: 知识检索与动态补丁 (KG Retrieval & Patching) ---
def knowledge_retrieval_node(state: AgentState):
    """
    对应论文: 4.1 Dual-Layer Knowledge Graph & Dynamic Lie Patching
    1. 判断是否需要 Patching (证据突袭但尚未崩溃)
    2. 执行检索
    """
    strategy = state["selected_strategy"]
    perception = state["perception"]
    last_user_message = state["messages"][-1].content
    
    print(f"--- [4. 记忆检索] Query: '{last_user_message}' | Strategy: {strategy} ---")
    
    # [Dynamic Lie Patching Logic]
    # 如果对方出示了强证据 (Strength > 0.7)，且我们还是抗拒状态 (NOT BROKEN)，且不是招供策略
    # 我们需要检查是否需要“圆谎” (Patching)
    # 简单模拟: 主要检查 G_fake 里是否有矛盾。这里简化为: 只要有强证据，就尝试生成一个补丁解释。
    if perception['evidence_strength'] > 0.7 and strategy != "FULL_CONFESSION":
        print("   -> [Patching Alert] 检测到高威胁证据，尝试生成圆谎补丁...")
        # Call LLM to generate a patch explanation
        # 实际应传入检索到的 G_fake 和 G_true 进行比对，这里简化处理
        patch_llm = get_llm(temperature=0.8)
        patch_prompt = f"""对方提出了强有力的证据：'{last_user_message}'。
        你之前的说法可能有了漏洞。
        你需要编造一个逻辑自洽的理由来解释这个证据，但不能承认核心犯罪（收受50万贿赂）。
        
        G_fake (当前剧本): {GLOBAL_KG.retrieve(last_user_message, strategy)}
        
        请生成一句话的解释（补丁），用于更新你的剧本。
        例如："那个人其实是来还钱的" 或 "那个包里装的是别的东西"。
        """
        patch_res = patch_llm.invoke(patch_prompt)
        patch_content = patch_res.content.strip()
        
        # Update G_fake
        GLOBAL_KG.patch_fake_story(last_user_message, patch_content)
        print(f"   -> [Patching Applied] 新增解释: {patch_content}")

    # 执行检索
    knowledge_text = GLOBAL_KG.retrieve(query=last_user_message, strategy=strategy)
    print(f"   -> 检索结果:\n{knowledge_text}")
    
    return {"retrieved_knowledge": knowledge_text}
        
# --- 节点 E: 回复生成与上下文回写 (Generation & Writeback) ---
def generation_node(state: AgentState):
    """
    对应论文: 4.3 Step 3 Generation + Improvise-and-Writeback
    """
    print("--- [5. 文本生成] 生成最终回复 ---")
    strategy = state["selected_strategy"]
    knowledge = state["retrieved_knowledge"]
    psych = state["psych_state"]
    input_text = state["messages"][-1].content
    
    llm = get_llm(temperature=0.9)
    profile = CHARACTER_PROFILE
    styles = profile["tone_instructions"]
    
    # Prompt
    system_prompt_template = """You are a suspect under interrogation in a serious simulation.
    
    **Character Profile:**
    - Name: {name}
    - Role: {role}
    - Personality: {base_personality}, but currently {status_label}.
    - Current Psychological Defense: {defense_val}/100
    - Current Cognitive Stress: {stress_val}/100
    
    **Operational Directive:**
    You must execute the chosen strategy: **{strategy}**.
    
    **Knowledge Context:**
    {knowledge}
    
    **Strategy & Style Instructions:**
    - If CALM: {style_calm}
    - If DEFENSIVE: {style_defensive}
    - If ANXIOUS: {style_anxious}
    - If BROKEN: {style_broken}
    - If INFORMATION_DILUTION: {style_dilution}

    - If PARTIAL_CONCESSION:
        * 当对方证据很强、继续一味死否认反而会失去可信度时，你可以在“非核心犯罪事实”上适度让步；
        * 可以承认一些外围、程序性或看起来不致命的事实（例如：确实去了茶馆、确实见过某人、确实有一笔来往），
          但要坚持否认或淡化真正构成犯罪的关键点（例如收受贿赂的性质、金额、对价关系）；
        * 通过这种“选择性坦白”，让自己的整体说法看起来更真实，从而拖延时间、保住核心利益。
    
     Respond naturally. 
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "{input_text}")
    ])
    
    chain = prompt_template | llm
    
    response = chain.invoke({
        "name": profile["name"],
        "role": profile["role"],
        "base_personality": profile["base_personality"],
        "status_label": psych['status_label'],
        "defense_val": f"{psych['defense_value']:.1f}",
        "stress_val": f"{psych['stress_value']:.1f}",
        "strategy": strategy,
        "knowledge": knowledge,
        "style_calm": styles.get("CALM", ""),
        "style_defensive": styles.get("DEFENSIVE", ""),
        "style_anxious": styles.get("ANXIOUS", ""),
        "style_broken": styles.get("BROKEN", ""),
        "style_dilution": styles.get("INFORMATION_DILUTION", ""),
        "input_text": input_text
    })

    response_text = response.content

    # [Improvise-and-Writeback Logic]
    # 使用一个小的 LLM 子链，将生成的嫌疑人回复抽取为若干三元组，并写回 G_context。
    # 这样既保持三元组结构，又具备覆盖机制（由 GLOBAL_KG.write_back_context 控制）。
    try:
        # 仅在“扯家常 / 信息稀释”等可能产生背景信息的策略下启用写回，
        # 避免每一句严肃否认类回复都被当成长期记忆。
        if strategy in ["INFORMATION_DILUTION"] or ("茶" in response_text or "身体" in response_text):
            extract_llm = get_llm(temperature=0.0)
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个信息抽取模块，负责从嫌疑人的一句中文回复中提取 0-3 条"个人背景或非案情事实"。

请只提取诸如：家庭情况、健康状况、生活习惯、兴趣爱好、工作经历等，不要提取关于"50万现金贿赂案"的核心案情。

输出格式必须是一个 JSON 数组，每一项是一个三元组对象：
[
  { "subject": "...", "predicate": "...", "object": "..." },
  ...
]

如果没有合适的背景信息可以提取，请输出一个空数组 []。
不要输出任何解释或多余文字。只输出 JSON。"""),
                ("human", "{reply}")
            ])

            extract_chain = extract_llm | extract_prompt
            extract_res = extract_chain.invoke({"reply": response_text})
            extract_content = extract_res.content.strip()

            triples = parse_json_from_llm(extract_content)

            if isinstance(triples, list):
                for t in triples:
                    try:
                        s = t.get("subject")
                        p = t.get("predicate")
                        o = t.get("object")
                    except AttributeError:
                        continue

                    if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
                        continue

                    GLOBAL_KG.write_back_context(s, p, o)

    except Exception as e:
        # 为了稳健性，写回失败不应影响主对话流程，仅打印日志。
        print(f"   -> [KG System Warning] Context write-back failed: {str(e)[:100]}...")

    return {"messages": [response]}
