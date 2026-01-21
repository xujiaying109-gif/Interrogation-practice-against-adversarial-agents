
import operator
from typing import Annotated, List, Dict, Union, Literal, Optional
from typing_extensions import TypedDict
from enum import Enum
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# ==========================================
# Data Structures
# ==========================================

class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object: str
    # tags: List[str] # 可选：用于标记这是核心犯罪事实还是外围信息

class GraphType(str, Enum):
    TRUE = "TRUE"
    FAKE = "FAKE"
    CONTEXT = "CONTEXT"

class DeceptiveOperatorType(str, Enum):
    DISTORTION = "DISTORTION"
    DELETION = "DELETION"
    FABRICATION = "FABRICATION"

class DeceptiveOperator(BaseModel):
    type: DeceptiveOperatorType
    target_triple: Optional[KnowledgeTriple] = None # For Distortion/Deletion
    new_tail: Optional[str] = None # For Distortion (new object)
    new_relation: Optional[str] = None # For Distortion (new predicate)
    fabricated_triple: Optional[KnowledgeTriple] = None # For Fabrication

class PsychologicalState(TypedDict):
    """
    心理状态向量
    对应论文: Dynamic Psychological State Machine (DPSM)
    """
    defense_value: float  # 心理防线 (0-100), 越低越容易招供
    stress_value: float   # 认知压力 (0-100), 越高越容易出错/语无伦次
    status_label: str     # 离散标签: CALM, DEFENSIVE, ANXIOUS, BROKEN

class AgentState(TypedDict):
    """
    DeepInquisitor 的全局状态
    """
    # 消息历史 (LangChain 标准格式)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 心理状态
    psych_state: PsychologicalState
    
    # 感知结果 (中间变量)
    perception: Dict
    
    # 对抗策略 (中间变量)
    selected_strategy: str
    
    # (Mock) 双层知识图谱的检索结果
    retrieved_knowledge: str 

class PerceptionOutput(BaseModel):
    """LLM 感知分析的结构化输出"""
    intent: Literal["EVIDENCE_PRESENTATION", "CHIT_CHAT", "PRESSURE"] = Field(
        description="审讯者的主要意图：出示证据、闲聊放松警惕、或施加压力"
    )
    evidence_strength: float = Field(
        description="如果意图是出示证据，证据的确凿程度(0.0-1.0)；如果是闲聊则为0",
        ge=0.0, le=1.0
    )
    is_trap: bool = Field(
        description="判断这是否是一个诱导性的陷阱问题"
    )
    analysis: str = Field(description="简短的内部分析理由")
