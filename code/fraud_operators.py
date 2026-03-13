from abc import ABC, abstractmethod
from typing import Dict, Any

class FraudOperator(ABC):
    """
    欺诈算子抽象基类 - 用于生成纪委审查场景下的特定对抗借口
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """算子名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """算子行为描述"""
        pass

    @abstractmethod
    def apply(self, target_node: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """
        应用算子生成借口
        :param target_node: 当前被逼问的知识图谱节点 (如某笔资金、某个人)
        :param context: 对话上下文与其他已暴露节点
        :return: 具体的对抗/伪装话术指导
        """
        pass


class LegalizationOperator(FraudOperator):
    """合法化包装算子 (民间借贷、投资分红等)"""
    @property
    def name(self) -> str:
        return "合法化包装算子"
        
    @property
    def description(self) -> str:
        return "将权钱交易包装成合法的民间借贷、亲属投资分红、正常的礼尚往来等。"

    def apply(self, target_node: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        item_name = target_node.get("amount", target_node.get("desc", "这笔资金/物品"))
        return f"坚称【{item_name}】是合法的民间经济往来（如正当借款、投资收益）。必须强调有借条或投资协议，且承担了市场风险，绝不承认是利益输送。"


class IsolationOperator(FraudOperator):
    """特定关系人隔离算子 (防火墙)"""
    @property
    def name(self) -> str:
        return "特定关系人隔离算子"
        
    @property
    def description(self) -> str:
        return "把责任全部推给特定关系人（亲属、司机等），声称自己毫不知情。"

    def apply(self, target_node: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        person_name = target_node.get("name", "特定关系人")
        return f"表现出痛心疾首，坚称是【{person_name}】背着自己狐假虎威、私下收受好处。强调自己平时工作太忙疏于管教，但绝对没有参与或指使。"


class CollectiveDecisionOperator(FraudOperator):
    """集体决策抗辩算子 (推卸责任)"""
    @property
    def name(self) -> str:
        return "集体决策抗辩算子"
        
    @property
    def description(self) -> str:
        return "以'集体研究'、'走程序'为借口，否认个人违规用权。"

    def apply(self, target_node: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        matter_name = target_node.get("desc", "这个项目/事项")
        return f"强调【{matter_name}】是经过局党组班子开会集体研究决定的，完全符合当时的政策和流程。自己只是作为一把手正常签字，没有任何个人私利夹带。"


class FraudOperatorFactory:
    _operators = {
        "合法化包装算子": LegalizationOperator(),
        "特定关系人隔离算子": IsolationOperator(),
        "集体决策抗辩算子": CollectiveDecisionOperator()
    }
    
    @classmethod
    def get_operator(cls, name: str) -> FraudOperator:
        """获取对应的欺诈算子，如果不存在返回 None"""
        return cls._operators.get(name)

    @classmethod
    def list_operators(cls):
        return list(cls._operators.keys())

