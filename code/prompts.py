"""
集中管理所有智能体的系统提示词和用户提示词模板 (Prompt Management)
此模块现在动态从 prompts/ 目录下的 Markdown 文件读取。
"""
import os

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")


def load_prompt(filename: str) -> str:
    """从 prompts 目录读取指定的 Markdown 提示词文件"""
    filepath = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ 警告: 找不到提示词文件 {filepath}")
        return ""


# ==========================================
# 审讯官 (Interrogator Agent) Prompts
# ==========================================
SYSTEM_PROMPT_INTERROGATOR = load_prompt("interrogator_system.md")
USER_PROMPT_INTERROGATOR = load_prompt("interrogator_user.md")
SYSTEM_PROMPT_INTELLIGENCE_FORGER = load_prompt("intel_system.md")
PROMPT_FABRICATE_EVIDENCE = load_prompt("fabricate_evidence.md")

# ==========================================
# 嫌疑人 (Suspect Agent) Prompts
# ==========================================
SYSTEM_PROMPT_SUSPECT = load_prompt("suspect_system.md")
SYSTEM_PROMPT_PERCEPTION = load_prompt("perception_system.md")

# ==========================================
# 裁判 (Referee Agent) Prompts
# ==========================================
SYSTEM_PROMPT_LOGIC_AUDITOR = load_prompt("logic_auditor_system.md")
USER_PROMPT_LOGIC_AUDITOR = load_prompt("logic_auditor_user.md")
SYSTEM_PROMPT_DATA_ANALYST = load_prompt("data_analyst_system.md")
PROMPT_SUMMARY_EVALUATION = load_prompt("summary_evaluation.md")
SYSTEM_PROMPT_LEAK_CHECK = load_prompt("leak_check_system.md")
