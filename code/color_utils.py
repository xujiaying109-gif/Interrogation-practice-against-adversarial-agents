"""
UI 样式工具 - 用于区分不同角色的控制台输出
"""
import os

# 启用 Windows 终端颜色支持
os.system('')


class Colors:
    # 角色色调
    POLICE = '\033[94m'  # 蓝色 (审讯官)
    SUSPECT = '\033[93m'  # 黄色 (嫌疑人)
    REFEREE = '\033[91m'  # 红色 (裁判)
    SYSTEM = '\033[96m'  # 青色 (系统/案件生成)
    TRUTH = '\033[92m'  # 绿色 (真相)

    # 样式
    BOLD = '\033[1m'
    DIM = '\033[2m'  # 暗淡 (用于思维链)
    RESET = '\033[0m'

    # 背景
    BG_POLICE = '\033[44m\033[97m'
    BG_SUSPECT = '\033[43m\033[97m'
    BG_REFEREE = '\033[41m\033[97m'


class UI:
    @staticmethod
    def print_role_header(role_name, icon, style=Colors.SYSTEM):
        print(f"\n{style}{'=' * 60}")
        print(f" {icon} {role_name}")
        print(f"{'=' * 60}{Colors.RESET}")

    @staticmethod
    def print_panel(content, title="", color=Colors.SYSTEM):
        print(f"{color}┌─ {title} {'─' * (55 - len(title))}")
        for line in content.split('\n'):
            print(f"│ {line}")
        print(f"└{'─' * 58}{Colors.RESET}")

    @staticmethod
    def print_thought(role_color, text):
        """打印思维链 (暗色显示)"""
        print(f"{role_color}{Colors.DIM}{text}{Colors.RESET}")

    @staticmethod
    def print_dialogue(role_color, name, text):
        """打印正式对话 (高亮显示)"""
        print(f"\n{role_color}{Colors.BOLD}█ {name}:{Colors.RESET} {text}")