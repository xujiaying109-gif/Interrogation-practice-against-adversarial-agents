"""
案件库 (Case Library)
动态从 cases/ 目录下读取所有 .json 案件配置。
已全部适配动态检索逻辑与 Tiered Case Data Structure (Fact -> Evidence_points)
"""
import os
import json
from log_utils import get_logger

logger = get_logger(__name__)

CASES = []

_base_dir = os.path.dirname(os.path.abspath(__file__))
_cases_dir = os.path.join(_base_dir, "cases")

if os.path.exists(_cases_dir):
    for filename in sorted(os.listdir(_cases_dir)):
        if filename.endswith(".json"):
            file_path = os.path.join(_cases_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    case_data = json.load(f)
                    CASES.append(case_data)
            except Exception as e:
                logger.error(f"Error loading case config {filename}: {e}")
else:
    logger.warning(f"Cases directory not found: {_cases_dir}")

def get_case_by_id(case_id: str):
    for c in CASES:
        if c.get("id") == case_id:
            return c
    return None
