from typing import TypedDict
from enum import Enum
class Triple(TypedDict, total=False):
    subject: str #主体
    predicate: str #动词
    object: str #客体
    meta:str #fake_story--欺骗算子 true_ground--地址/背景/NONE
    evidence_strength: float #证据强度
    narrative: str #叙述性解释
 # 欺骗算子枚举
class DeceptionOperator(str, Enum):
    DISTORT = "DISTORT"         # 扭曲事实
    OMIT = "OMIT"               # 删减信息
    FABRICATE = "FABRICATE"     # 捏造信息
    DENY = "DENY"               # 直接否认
    RATIONALIZE = "RATIONALIZE" # 合理化解释
    NONE = "NONE"               # 无（用于事实层）
