"""
API 客户端
"""
import time
from typing import List, Dict, Optional
import os
import requests
from log_utils import get_logger

logger = get_logger(__name__)

class QianwenAPIClient:
    def __init__(self, base_url, model_name, api_key):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key

    def check_health(self):
        """实时连通性检查，并返回具体的错误类型供 UI 显示"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                return True, "连接正常"
            elif response.status_code == 401:
                return False, "API Key 无效 (401)"
            elif response.status_code == 404:
                return False, "接口地址/模型名错误 (404)"
            else:
                return False, f"HTTP 错误: {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "连接超时 (请检查网络)"
        except Exception as e:
            return False, f"异常: {str(e)}"

    def generate_response(self, system_prompt, user_input, temperature=0.7, max_tokens=500):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=80)
            if response.status_code == 200:
                result = response.json()
                if "choices" in result:
                    return result["choices"][0]["message"]["content"]
                return str(result)
            else:
                return f"API Error: {response.status_code}"
        except Exception as e:
            logger.error(f"API 请求发生错误: {e}")
            return f"Error: {str(e)}"