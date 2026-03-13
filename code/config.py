"""
配置文件 (增加审讯方情报配置)
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

@dataclass
class AgentConfig:
    # --- API 配置 ---
    USE_MOCK_LLM: bool = False
    API_BASE_URL: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "http://222.171.219.26:20001/v1"))
    MODEL_NAME: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507"))
    API_KEY: str = field(default_factory=lambda: os.getenv("API_KEY", ""))

    # --- LLM 参数 ---
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500

    # --- 嫌疑人策略阈值 ---
    FULL_CONFESSION_DEFENSE: float = 5.0
    FULL_CONFESSION_STRESS: float = 95.0
    DIRECT_DENIAL_DEFENSE: float = 70.0
    FEIGN_IGNORANCE_DEFENSE_MIN: float = 40.0
    FEIGN_IGNORANCE_DEFENSE_MAX: float = 80.0
    FEIGN_IGNORANCE_STRESS_MIN: float = 0.0
    FEIGN_IGNORANCE_STRESS_MAX: float = 60.0
    HIGH_EVIDENCE_THRESHOLD: float = 0.6

    # --- [新增] 审讯官情报配置 ---
    # 真实证据比例 (例如 0.6 表示掌握的线索中 60% 是准确的)
    POLICE_EVIDENCE_TRUTH_RATIO: float = 0.6
    # 虚假/错误情报比例 (例如 0.4 表示 40% 的线索是有偏差的，如时间地点错误)
    POLICE_EVIDENCE_NOISE_RATIO: float = 0.4

    # --- 默认嫌疑人 ---
    DEFAULT_SUSPECT = {
        "name": "张局长",
        "personality": "arrogant",
        "speaking_style": "formal"
    }

config = AgentConfig()