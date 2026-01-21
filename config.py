
import os
from langchain_openai import ChatOpenAI

# ==========================================
# 0. 配置与模型工厂 (Configuration)
# ==========================================

# 配置自定义兼容 OpenAI 的接口参数
# 您可以直接在这里修改字符串，或者在环境变量中设置
# 示例 URL: "http://localhost:8000/v1" (vLLM/Ollama) 或 您的中转服务地址
API_BASE_URL = os.getenv("OPENAI_API_BASE", "http://222.171.219.26:20001/v1")
API_KEY = os.getenv("OPENAI_API_KEY","gpustack_5cfc0c1861d1e5e9_0365bc79b88d947309d93c6ae33b00f4")
# 您的模型名称，需与您自建服务中的模型 ID 一致 (例如 "llama3-70b", "qwen-72b" 等)
# 如果是完全兼容 GPT-4 的代理，保留 "gpt-4o" 即可
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

def get_llm(temperature=0.7):
    """获取配置好的 LLM 实例 (支持自定义 OpenAI 兼容接口)"""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
