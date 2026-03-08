import os
from datetime import datetime
from typing import Dict, List, Any
from log_utils import get_logger

logger = get_logger(__name__)

def save_interrogation_report(gs: Dict, report: Dict) -> str:
    """
    自动导出审讯全过程到 Markdown 报告。
    """
    save_dir = "saved_reports"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_title = "AI_Interrogation"
    # 尝试提取案件名
    if "messages" in gs and len(gs["messages"]) > 0:
         case_title = "case_record"

    filename = f"{save_dir}/{timestamp}_{case_title}.md"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# 🕵️ DeepInquisitor 审讯报告 (UI版)\n")
            f.write(f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **运行轮次**: {gs.get('current_round', 0)} / {gs.get('max_rounds', 0)}\n\n")

            # 1. 裁判评估
            f.write("## 1. 裁判结案评估\n```text\n")
            if 'summary' in report:
                f.write(report['summary'])
            f.write("\n```\n\n")
            
            f.write(f"- 🏆 获胜方: {report.get('winner', '未知')}\n")
            f.write(f"- 📊 逻辑分: {report.get('logic_score', '0')}\n")
            f.write(f"- 📈 泄露率: {report.get('leak_rate', '0%')}\n\n")

            # 2. 认知差构建记录
            f.write("## 2. 警方情报池 (认知差线索)\n")
            if "intel_logs" in gs:
                for idx, log in enumerate(gs["intel_logs"]):
                    status = "真实" if log['is_real'] else "虚假"
                    f.write(f"### [{status}] 线索 {idx+1}\n")
                    f.write(f"- **警方视角**: {log['p_view']}\n")
                    f.write(f"- **实际真相**: {log['s_view']}\n\n")

            # 3. 对话全记录
            f.write("## 3. 详细审讯记录\n")
            for m in gs["messages"]:
                if m["role"] == "police":
                    f.write(f"### 第 {m.get('round', '?')} 轮\n")
                    f.write(f"**👮 审讯官**: {m['content']}\n\n")
                elif m["role"] == "suspect":
                    f.write(f"**🦊 嫌疑人**: {m['content']}\n\n")
                    psych = m.get('psych', {})
                    f.write(f"> 状态监控: 防御 {psych.get('def', 0):.1f} | 压力 {psych.get('str', 0):.1f}\n")
                    ref = m.get('ref', {})
                    if ref.get("conflict"):
                        f.write(f"\n> ⚠️ **裁判介入 (矛盾点)**: {ref['conflict']}\n")
                    f.write("\n---\n")

        logger.info(f"✅ 审讯记录已自动导出至: {filename}")
        return filename

    except Exception as e:
        logger.error(f"❌ 保存报告失败: {str(e)}")
        return ""
