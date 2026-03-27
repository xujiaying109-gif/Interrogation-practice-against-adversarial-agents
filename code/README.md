# 🕵️‍♂️ DeepInquisitor 审讯对抗智能体

DeepInquisitor 是一个基于 LangGraph 和大型语言模型 (LLM) 构建的多智能体 (Multi-Agent) 审讯博弈模拟系统。该系统允许玩家扮演主审官，通过心理攻势与策略提问，与具备独立人格和记忆的 AI 嫌疑人进行实时推理对抗。

---

## 🚀 核心架构与原理

系统通过多个独立协同的 Agent 构成对抗循环，底层不再是简单的 While 循环，而是受 `LangGraph` 原生 `StateGraph` 与 `interrupt_after` 断点机制强力驱动，支持 Streamlit 进行 Web 渲染交互。核心包含三方势力：
1. **审讯官 (Red Team)**：研读警方案卷与线索，实施压力审讯，并通过内置逻辑实时分析嫌疑人破绽。
2. **嫌疑人 (Blue Team)**：拥有独立“心智（Stress & Defense）”、基于知识图谱 (Knowledge Graph) 维护的长期事件记忆。会根据当前心情与受到证据的恐吓程度，动态挑选撒谎、反抗或逃避策略。
3. **逻辑裁判 (Referee)**：作为全局视角的规则维护者，实时盘查嫌疑人的供词是否存在事实性逻辑矛盾（如时间线倒错、核心要素改口），并结合泄露率最终判定某方胜出。

*(更详细的演进与功能更新，请参阅本项目的 [CHANGELOG.md](CHANGELOG.md))*

---

## 📂 项目文件结构与职责

```text
├── .env                # (需新建) 本地环境变量，存储 API_KEY 等敏感信息
├── .env.example        # 环境变量模板参考
├── pyproject.toml      # Poetry 项目依赖管理配置文件
├── README.md           # 项目说明文档
│
├── app.py              # 🖥️ 主线 UI 入口，基于 Streamlit 实现界面与状态管理
├── workflow.py         # ⚙️ LangGraph 核心图谱工厂，利用 `build_langgraph_app` 编译状态机
│
├── config.py           # 🎛️ 全局配置类 (AgentConfig)，统一管理 API 接口地址、模型名、心理防御等阈值系数
├── api_client.py       # 🔌 统一的 LLM (如千问) 网络请求客户端封装
├── log_utils.py        # 📋 日志中心：使用 loguru 处理控制台彩打与本地 `logs/` 归档记录
├── export_utils.py     # 💾 数据导出：审讯结束后自动生成 Markdown 复盘报告存放在 `saved_reports/`
│
├── case_library.py     # 📚 案件预设库，存储剧本的初始真相事件、干扰事件以及核心关键词
├── prompts.py          # 🗣️ 提示词装载器，动态从 prompts/ 目录加载各个角色的指令模板
├── prompts/            # 📝 Markdown 格式的系统提示词文件夹，便于直观修改所有智能体口吻
│
├── interrogator.py     # 🕵️ 审讯官智能体：负责理解案卷、生成警方情报网（包括造假干扰项）、以及选择审讯策略
├── suspect_agent.py    # 👤 嫌疑人智能体顶层：组装感知、心理、策略和生成等多个子模块
├── referee.py          # ⚖️ 裁判智能体：实时进行多轮口供对比以检测矛盾，并基于漏词率生成结案报告
│
├── personality.py      # 🎭 [嫌疑人子模块] 策略模式性格抽象：利用多态管理“傲慢/紧张/冷静”的承压计算和发言倾向
├── perception.py       # 👁️ [嫌疑人子模块] 感知器：分析警察提问的意图、威胁强度与是否设下陷阱
├── psych_state.py      # 💓 [嫌疑人子模块] 心理状态机：接受性格实例，计算每回合防线血量(Defense)与压力(Stress)起伏
├── strategy.py         # ♟️ [嫌疑人子模块] 策略中心：结合感知与心理现状，决策采用强硬辩解还是示弱欺瞒
├── event_kg_module.py  # 🧠 [嫌疑人子模块] 知识图谱记忆引擎：存储底层事实，并在发言前检索近期对话与既定谎言以防金鱼记忆崩溃
├── generator.py        # 🎙️ [嫌疑人子模块] 语言生成器：严格遵循纪律与口供约束，并根据决策层生成的策略最终产出文段
│
├── color_utils.py      # 🎨 终端颜色美化工具，用于 CLI 日志高亮
└── tests/              # 🧪 pytest 单元测试目录 (如 test_kg_module.py, test_psych_state.py)
```

---

## 🛠 快速上手指南

### 1. 环境安装
项目使用 `poetry` 进行严格的包依赖管理。请确保本机已安装 Python 3.11+ 和 Poetry。

```bash
# 进入项目根目录执行
poetry install
```

### 2. 配置环境变量
复制模板配置并填入您自己的可用 API 凭证：
```bash
cp .env.example .env
# 之后编辑 .env，将 API_KEY 填入
```

### 3. 运行启动
我们推荐使用可视化的 Web 界面来沉浸式体验与查看属性数值：
```bash
poetry run streamlit run app.py
```
若希望在无头环境下进行纯 CLI 终端运行和测试基于图的 workflow：
```bash
poetry run python workflow.py
```

---

## 📈 核心特性说明

* **防线机制 (Defense Breakdown)**
嫌疑人的防线 (Defense) 并非匀速下降。面对具有致命逻辑杀伤力的实质证据，防线会产生暴击削减；若审讯员的问题绵软无力，嚣张性格的嫌疑人甚至会让 Defense 回弹。
* **反遗忘检索网络 (Anti-Goldfish Memory)**
嫌疑人在抛出一项伪造设定（例如“我昨晚在睡觉”）后，`EventRetrievalEngine` 会显式通过提示词锁定将其织入未来的对话 Context 中，防止发生低级“上一轮睡觉，这一轮在外吃饭”的错乱口供，逼迫玩家使用更加高级的侦讯刺探技巧。
* **动态配置中心**
所有的 prompt 提示词都被高度解耦到 `prompts/*.md` 目录下。非程序员人员也可以随时参与调整 AI 的语气激烈度与回答字数。
