
import json
from typing import Any

def parse_json_from_llm(content: str) -> Any:
    """
    Parses JSON content from an LLM response.
    Handles standard JSON and markdown-wrapped JSON (```json ... ```).
    """
    content = content.strip()
    
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
        
    if content.endswith("```"):
        content = content[:-3]
        
    content = content.strip()
    
    return json.loads(content)
