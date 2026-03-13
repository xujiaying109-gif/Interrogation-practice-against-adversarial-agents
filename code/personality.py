"""
嫌疑人人格体系 (Strategy Pattern)
定义不同的性格特征，影响抗压乘数与谎言策略倾向。
"""
from abc import ABC, abstractmethod

class BasePersonality(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def initial_defense(self) -> float:
        pass

    @property
    @abstractmethod
    def initial_stress(self) -> float:
        pass
        
    @property
    @abstractmethod
    def volatility(self) -> float:
        """情绪波动范围 (±百分比)"""
        pass

    @property
    @abstractmethod
    def core_traits(self) -> str:
        """核心特质与行为表现，用于注入大模型System Prompt"""
        pass

    @property
    @abstractmethod
    def catchphrases(self) -> list[str]:
        """常见话术，用于约束大模型的话语风格"""
        pass

    @property
    @abstractmethod
    def reaction_patterns(self) -> list[str]:
        """反应模式（战术小动作），决定角色的现场微观战术选择"""
        pass

    @abstractmethod
    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        """
        计算受到不同审讯战术时的真实防线扣减量
        :param base_drop: 预计的基础扣减量
        :param strat_name: 警方策略 (例如：心理施压，出示证据，缓和安抚等)
        """
        pass

    @abstractmethod
    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        """
        计算受到不同审讯战术时的真实压力上升量
        :param base_rise: 预计的基础上升量
        :param strat_name: 警方策略 
        """
        pass

    @abstractmethod
    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        pass

    @abstractmethod
    def get_verbosity_rules(self, stress_value: float) -> str:
        """根据压力值返回该性格特点的回复长度与语速指令"""
        pass


class ArrogantPersonality(BasePersonality):
    """傲慢型：对口头施压免疫度高，但面对铁证容易破防"""
    name = "傲慢型"
    initial_defense = 70.0
    initial_stress = 30.0
    volatility = 0.1

    @property
    def core_traits(self) -> str:
        return "自视甚高，权力优越感强。态度傲慢嚣张，不把谈话人员放在眼里，言语极具挑衅性。认为自己背景硬，不会被深究。"

    @property
    def catchphrases(self) -> list[str]:
        return ["大家都这么做，凭什么只查我。", "我上面有人，你们最好想清楚。", "你去叫你们领导来和我谈。"]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "直接反驳：居高临下地否定主审官的假设。", "max_stress": 60, "trigger_strategy": ["正常询问", "连续追问"]},
            {"pattern": "威胁恐吓：暗示自己有靠山，警告主审官不要惹事。", "max_stress": 80, "trigger_strategy": ["心理施压"]},
            {"pattern": "转移责任：轻描淡写地把责任推给下属。", "max_stress": 100, "trigger_strategy": ["出示证据", "连续追问"]},
            {"pattern": "不屑一顾：冷笑，对主审官的问题表现出极度的轻蔑。", "max_stress": 50, "trigger_strategy": ["正常询问", "缓和安抚"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        if stress_value < 50:
            return "【字数建议】：控制在 100 字以内，语气强势简短，体现不屑多说的傲慢。"
        return "【字数建议】：控制在 250 字以内，由于感到了压力，开始用强调自己功劳、背景的话来找补和掩饰。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.8, "心理施压": 0.5, "连续追问": 0.8, "缓和安抚": 0.1}
        return base_drop * effect_map.get(strat_name, 0.8)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.5, "心理施压": 0.7, "连续追问": 1.0, "缓和安抚": 0.2}
        return base_rise * effect_map.get(strat_name, 0.8)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 10:  # 只有极低血量才招
            return "full_confession"
        if intent == "pressure" or defense_value > 40:
            return "direct_denial"  # 习惯性强硬对抗
        if evidence_strength > 0.6:
            return "rationalization"
        return "feign_ignorance"


class NervousPersonality(BasePersonality):
    """紧张焦虑型：受压立崩，风吹草动都会加剧紧张"""
    name = "紧张焦虑型"
    initial_defense = 80.0
    initial_stress = 20.0
    volatility = 0.2

    @property
    def core_traits(self) -> str:
        return "心理承受能力弱，极度缺乏安全感。表情、肢体紧张感明显，眼神慌乱，语无伦次。会在交代与反悔之间反复横跳。"

    @property
    def catchphrases(self) -> list[str]:
        return ["我真的不知道该怎么办了，我太害怕了。", "如果我说了能帮我保密吗？", "我记错了，刚才说的不对，我再想想。"]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "语无伦次：极度慌乱，说话颠三倒四，前后矛盾。", "max_stress": 100, "trigger_strategy": ["心理施压", "出示证据", "连续追问"]},
            {"pattern": "反复推翻：刚才还在否认，被追问后马上承认，然后又试图改口。", "max_stress": 100, "trigger_strategy": ["连续追问"]},
            {"pattern": "哀求宽恕：一边哭泣一边恳求主审官放过自己。", "max_stress": 100, "trigger_strategy": ["心理施压", "出示证据"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        if stress_value < 60:
            return "【字数建议】：控制在 150 字左右，语气紧张，多使用犹豫和重复词汇。"
        return "【字数建议】：控制在 300 字左右，极度恐慌下语无伦次，把毫不相干的事情也全盘托出以祈求宽大。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"心理施压": 1.6, "出示证据": 1.3, "连续追问": 1.4, "缓和安抚": 0.5}
        return base_drop * effect_map.get(strat_name, 1.0)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"心理施压": 1.8, "出示证据": 1.2, "连续追问": 1.5, "缓和安抚": -0.8} # 安抚能显著降压
        return base_rise * effect_map.get(strat_name, 1.1)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 25 or stress_value > 85: # 很容易就招供了
            return "full_confession"
        if evidence_strength > 0.4:
            return "rationalization" # 容易怂而找借口
        if is_trap or intent in ["time_inquiry", "location_inquiry"]:
            return "feign_ignorance"
        return "feign_ignorance" # 偏好回避


class CalmPersonality(BasePersonality):
    """冷静型：情绪稳定，油盐不进"""
    name = "冷静型"
    initial_defense = 75.0
    initial_stress = 25.0
    volatility = 0.05

    @property
    def core_traits(self) -> str:
        return "情绪极其稳定，心理素质高。回答问题滴水不漏，不见兔子不撒鹰，只有在核心证据面前才会动摇。"

    @property
    def catchphrases(self) -> list[str]:
        return ["我不清楚你说的意思。", "这种假设性的问题我无法回答。", "证据呢？拿出来我看看。"]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "冷漠应对：用极其简短、毫无波澜的语气回答。", "max_stress": 60, "trigger_strategy": ["正常询问", "心理施压", "缓和安抚"]},
            {"pattern": "要求证据：不顺着主审官的逻辑走，反过来索要实质性的物证。", "max_stress": 80, "trigger_strategy": ["心理施压", "连续追问"]},
            {"pattern": "逻辑反问：敏锐地指出主审官问话中的逻辑漏洞。", "max_stress": 70, "trigger_strategy": ["连续追问", "出示证据"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        if stress_value < 70:
            return "【字数建议】：控制在 50 字以内，极其简短、冷漠，绝不多说一个废话。"
        return "【字数建议】：控制在 150 字以内，即便有防御压力依然试图保持逻辑严密地进行少量辩驳。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.2, "心理施压": 0.7, "连续追问": 0.9, "缓和安抚": 0.3}
        return base_drop * effect_map.get(strat_name, 0.8)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.2, "心理施压": 0.8, "连续追问": 1.0, "缓和安抚": 0.4}
        return base_rise * effect_map.get(strat_name, 0.8)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 5: # 极难被破防
            return "full_confession"
        if evidence_strength > 0.7:
            return "rationalization"
        if is_trap:
            return "feign_ignorance"
        return "direct_denial"

class CalculatingPersonality(BasePersonality):
    """博弈计算型：精明理性，极度现实，功利心突出"""
    name = "博弈计算型"
    initial_defense = 80.0
    initial_stress = 20.0
    volatility = 0.05

    @property
    def core_traits(self) -> str:
        return (
            "精明理性，极度现实，功利心突出。将谈话视为“主动主导的利益交换”博弈。"
            "回答谨慎克制，字斟句酌。初期采取“挤牙膏”式回答，试探边界；后期会放出局部、次要的交代换取信任，以此掩盖核心违纪违法事实。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "你们先说说掌握的情况，我也好如实交代。",
            "这事我有责任，但主要不是我，我只是配合执行。",
            "我这样算不算主动交代？怎样算从轻处理？",
            "我知道的都说说，但我不知道的，也不能乱讲。"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "试探底牌：反问主审官以探寻纪委到底掌握了多少证据。", "max_stress": 60, "trigger_strategy": ["正常询问", "心理施压"]},
            {"pattern": "弃卒保车：主动交代一些无关痛痒的外围违规事实，试图表现出“配合”来蒙混过关。", "max_stress": 90, "trigger_strategy": ["出示证据", "连续追问"]},
            {"pattern": "谈心交易：试图和主审官谈条件，询问主动交代的量纪标准。", "max_stress": 100, "trigger_strategy": ["缓和安抚", "出示证据"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        if stress_value < 50:
            return "【字数建议】：控制在 100 字以内，字斟句酌，非常谨慎。"
        return "【字数建议】：控制在 250 字左右，开始像谈生意一样，抛出一些外围事实试图换取信任。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.2, "心理施压": 0.2, "连续追问": 0.8, "缓和安抚": 0.5}
        return base_drop * effect_map.get(strat_name, 0.6)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.3, "心理施压": 0.5, "连续追问": 0.8, "缓和安抚": -0.2}
        return base_rise * effect_map.get(strat_name, 0.7)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 15:
            return "full_confession"
        if evidence_strength > 0.6:
            return "rationalization" # 局部交代换取信任
        if intent == "pressure":
            return "direct_denial"
        return "feign_ignorance" # 初期模糊化回应


class AnxiousPersonality(BasePersonality):
    """焦虑脆弱型：心理承受能力弱，情绪敏感脆弱，抗压能力差"""
    name = "焦虑脆弱型"
    initial_defense = 60.0
    initial_stress = 40.0
    volatility = 0.25

    @property
    def core_traits(self) -> str:
        return (
            "心理承受能力极度脆弱，易被纪律压力、证据出示击垮。处于“恐慌—犹豫—反悔”的循环中。"
            "语气不稳定，严重时语无伦次、结巴甚至大哭。供述缺乏逻辑，经常前后矛盾或者反复否认自己刚说过的话。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "会不会连累我的家人？他们不知道这件事。",
            "我真的记不清了，当时太慌乱了。",
            "你别说是我说的，我出去还得混呢。",
            "我真的不知道该怎么办了，我太害怕了。"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "精神崩溃：情绪失控，大声哭泣或身体发抖。", "max_stress": 100, "trigger_strategy": ["心理施压", "出示证据"]},
            {"pattern": "全盘招供：心理防线被击穿，毫无保留地交代一切，甚至连带出其他不相关的事情。", "max_stress": 100, "trigger_strategy": ["连续追问", "出示证据"]},
            {"pattern": "过度脑补：把主审官的一句普通问话解读为严重的警告，陷入自我恐慌。", "max_stress": 80, "trigger_strategy": ["正常询问", "缓和安抚"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        return "【字数建议】：控制在 300 字左右，情绪非常不稳定，夹杂大量无意义的担忧、反问和哭诉。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"心理施压": 1.8, "出示证据": 1.5, "连续追问": 1.6, "缓和安抚": 0.4}
        return base_drop * effect_map.get(strat_name, 1.2)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"心理施压": 2.0, "出示证据": 1.5, "连续追问": 1.8, "缓和安抚": -1.5}
        return base_rise * effect_map.get(strat_name, 1.4)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 35 or stress_value > 80:
            return "full_confession"
        if evidence_strength > 0.3:
            return "rationalization"
        return "feign_ignorance" # 犹豫或敷衍


class DefiantPersonality(BasePersonality):
    """狂妄抵触型：自视劳苦功高，优越感极强，态度傲慢嚣张"""
    name = "狂妄抵触型"
    initial_defense = 90.0
    initial_stress = 10.0
    volatility = 0.15

    @property
    def core_traits(self) -> str:
        return (
            "自视劳苦功高，长期处于高位导致优越感极强，对调查人员怀有敌意。认为自己有靠山、背景硬。"
            "态度傲慢嚣张，充满挑衅甚至反驳指责警务人员，强行辩解为“行业惯例”、“工作需要”。极难沟通。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "有本事你们就直接处理我，我不怕。",
            "我上边有人，你们最好识相点，别给自己找麻烦。",
            "那么多人都那样，你们怎么不一个个查呢？",
            "那都是下属擅自做的，我不知情。"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "直接反驳：不绕弯子，毫无顾忌地直接否定对方的假设。", "max_stress": 70, "trigger_strategy": ["正常询问", "缓和安抚"]},
            {"pattern": "情绪爆发：被激怒后大声咆哮，指责纪委办案不公。", "max_stress": 100, "trigger_strategy": ["心理施压", "连续追问", "出示证据"]},
            {"pattern": "高傲蔑视：使用极度轻蔑和挑衅的用词，暗示办案人员级别不够。", "max_stress": 60, "trigger_strategy": ["正常询问", "心理施压"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        if stress_value < 60:
            return "【字数建议】：控制在 80 字以内，像首长训斥人一样狂妄短促。"
        return "【字数建议】：控制在 200 字左右，情绪激动愤怒，用大段的长篇大论指责、咆哮主审官。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        # 对施压甚至可能增加防御(负值或极低值)
        effect_map = {"心理施压": 0.1, "出示证据": 1.2, "连续追问": 0.5, "缓和安抚": 0.2}
        return base_drop * effect_map.get(strat_name, 0.5)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"心理施压": 0.8, "出示证据": 1.2, "连续追问": 1.5, "缓和安抚": 0.1}
        return base_rise * effect_map.get(strat_name, 0.6)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 10 and evidence_strength > 0.8:
            return "full_confession"
        return "direct_denial" # 习惯性强硬对抗、转移责任


class SlickPersonality(BasePersonality):
    """圆滑伪装型：情商高，善于察言观色，擅长表演和塑造“配合”人设"""
    name = "圆滑伪装型"
    initial_defense = 75.0
    initial_stress = 30.0
    volatility = 0.1

    @property
    def core_traits(self) -> str:
        return (
            "情商高，极其善于察言观色和表演。表面态度谦和诚恳、积极配合，实则规避核心问题。"
            "极其擅长用“正确的废话”拖延时间、转移焦点，对实质性违纪违法问题避重就轻，用大道理掩盖小问题。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "我一定全力配合，绝不隐瞒任何问题。",
            "这事确实有我的责任，我反思过很多次，以后一定整改。",
            "我不是主要负责人，具体情况我不太清楚，我只是听说过。",
            "钱是借的，当时太忙，忘了打借条，后续一直没来得及补。"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "假意配合：态度极其端正，满口答应，但回答全是空泛的套话和废话。", "max_stress": 80, "trigger_strategy": ["正常询问", "缓和安抚"]},
            {"pattern": "偷换概念：把严重的权钱交易偷换成“人情往来”或“工作失误”。", "max_stress": 100, "trigger_strategy": ["出示证据", "连续追问"]},
            {"pattern": "打太极拳：面对核心问题顾左右而言他，把话题引向无关的琐碎生活细节。", "max_stress": 90, "trigger_strategy": ["连续追问", "心理施压"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        return "【字数建议】：非常能扯（300 - 400字），满嘴“正确的废话”、“假大空的官话”，绕圈子拖延时间，避重就轻。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"连续追问": 1.4, "出示证据": 1.5, "心理施压": 0.6, "缓和安抚": 0.8}
        return base_drop * effect_map.get(strat_name, 0.9)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"连续追问": 1.3, "出示证据": 1.2, "心理施压": 0.8, "缓和安抚": -0.5}
        return base_rise * effect_map.get(strat_name, 0.9)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 15:
            return "full_confession"
        if evidence_strength > 0.5:
            return "rationalization" # 避重就轻，用大道理掩盖小问题
        return "feign_ignorance" # 转移话题，用废话拖延


class SilentPersonality(BasePersonality):
    """沉默戒备型：心理防线极其坚固，性格内向隐忍，信奉“言多必失”"""
    name = "沉默戒备型"
    initial_defense = 95.0
    initial_stress = 15.0
    volatility = 0.02

    @property
    def core_traits(self) -> str:
        return (
            "极度固执，防线异常坚固，信奉“言多必失”。无论遭受何种压力和语气质问，都使用极简语言作为防御。"
            "全程平淡、不发声、眼神回避。有极强的耐心，不轻易崩溃，内心活动剧烈但外表像一头死猪。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "不知道。",
            "不清楚。",
            "记不得了。",
            "无可奉告。"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "沉默对抗：对长篇大论的提问仅用寥寥几个字回应。", "max_stress": 100, "trigger_strategy": ["心理施压", "出示证据", "连续追问", "缓和安抚", "正常询问"]},
            {"pattern": "机械重复：无论怎么换着方式问，都只重复同一句“不知道”。", "max_stress": 100, "trigger_strategy": ["心理施压", "连续追问"]},
            {"pattern": "闭口不答：不仅不说话，还伴随闭眼、转头等拒绝交流的肢体动作。", "max_stress": 100, "trigger_strategy": ["连续追问"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        if stress_value < 85:
            return "【字数建议】：极度简省，绝对不要超过 15 个字。几乎不发声，甚至只用“嗯”、“没”等单字敷衍。"
        return "【字数建议】：即便被迫开口，也绝不长篇大论，控制在 50 字以内。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.5, "心理施压": 0.1, "连续追问": 0.2, "缓和安抚": 0.8}
        return base_drop * effect_map.get(strat_name, 0.3)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.2, "心理施压": 0.3, "连续追问": 0.5, "缓和安抚": -0.2}
        return base_rise * effect_map.get(strat_name, 0.4)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 10 and evidence_strength > 0.8:
            return "full_confession"
        return "feign_ignorance" # 表现为拒绝交流或极简回答


class LoyalPersonality(BasePersonality):
    """江湖义气型：看重“人情”，主动承揽罪责，保护他人"""
    name = "江湖义气型"
    initial_defense = 85.0
    initial_stress = 30.0
    volatility = 0.12

    @property
    def core_traits(self) -> str:
        return (
            "有独特的地下伦理观，将“讲义气”“不出卖朋友”视作绝对底线，甚至超越法律。"
            "非常在意圈内声誉。可能会极其坚定地一人揽下所有罪责，包庇同案犯，对法制教育非常反感。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "出卖朋友的事，我干不出来，我做人有底线。",
            "要处理就处理我，别找其他人，他们全都不知情。",
            "这是我和他之间的事，用不着你们管。",
            "一人做事一人当！"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "大包大揽：坚决把所有的罪名都揽到自己一个人头上，咬死没有同谋。", "max_stress": 100, "trigger_strategy": ["由于证据", "连续追问", "心理施压"]},
            {"pattern": "抗拒诱导：面对“宽大处理”的诱惑不为所动，甚至出言讥讽。", "max_stress": 80, "trigger_strategy": ["缓和安抚"]},
            {"pattern": "极度护短：一旦主审官提到要调查其同伙或家属，情绪立刻变得激动并发出警告。", "max_stress": 100, "trigger_strategy": ["心理施压", "出示证据", "连续追问"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        return "【字数建议】：控制在 100 - 150 字左右，语气江湖气重，短促而决绝，一人兜揽到底。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.4, "心理施压": 0.5, "连续追问": 0.8, "缓和安抚": 1.0}
        return base_drop * effect_map.get(strat_name, 0.7)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.3, "心理施压": 0.8, "连续追问": 1.0, "缓和安抚": -0.5}
        return base_rise * effect_map.get(strat_name, 0.8)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 20: # 只有极低防御时才会交代(可能依然不攀扯)
            return "full_confession" 
        if evidence_strength > 0.5:
            return "rationalization"
        return "direct_denial" # 强硬否认或一人扛下


class OpportunisticPersonality(BasePersonality):
    """求生投机型：功利心外露，缺乏底线，急于保命减罚"""
    name = "求生投机型"
    initial_defense = 50.0
    initial_stress = 50.0
    volatility = 0.3

    @property
    def core_traits(self) -> str:
        return (
            "没有任何底线和原则，核心诉求是“紧急保命、减罚”。"
            "非常容易为了争取立功而胡乱攀咬他人、提供虚假线索、夸大自己的配合度。极度卑躬屈膝，见风使舵。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "我全交代！立功能减轻处分吗？",
            "我还有别人的线索要举报，能不能换我宽大处理？",
            "我承认错误，请对我网开一面，都是被他们带坏的！",
            "我这样算不算主动交代啊？"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "急于立功：主动抛出一些惊人的但不一定真实的“大瓜”，试图转移视线并邀功。", "max_stress": 100, "trigger_strategy": ["心理施压", "连续追问"]},
            {"pattern": "卑躬屈膝：极度讨好主审官，使用夸张的谄媚词汇。", "max_stress": 100, "trigger_strategy": ["缓和安抚", "正常询问"]},
            {"pattern": "疯狂甩锅：把所有的主观恶性都推给同事或领导，把自己塑造成无辜的受害者。", "max_stress": 100, "trigger_strategy": ["出示证据", "连续追问"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        return "【字数建议】：控制在 250 - 350 字左右，由于极度想邀功保命，会像倒豆子一样说一大堆话，卑微且急促。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"出示证据": 2.0, "心理施压": 1.5, "连续追问": 1.2, "缓和安抚": 0.5}
        return base_drop * effect_map.get(strat_name, 1.2)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.8, "心理施压": 1.6, "连续追问": 1.4, "缓和安抚": -1.0}
        return base_rise * effect_map.get(strat_name, 1.2)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 40 or evidence_strength > 0.4:
            return "full_confession" # 急于交代，甚至可能乱攀咬
        return "rationalization" # 试图通过辩解答应对策


class RepentantPersonality(BasePersonality):
    """诚实悔过型：良知未泯，愿意彻底交代，真诚悔过"""
    name = "诚实悔过型"
    initial_defense = 30.0
    initial_stress = 70.0
    volatility = 0.1

    @property
    def core_traits(self) -> str:
        return (
            "良知未泯，极其懊悔。愿意彻底如实交代每一项细节。有强烈的愧疚感和负罪感。"
            "甚至会主动交代警方并未掌握的细节内容。不推诿不逃避，表现得非常痛心疾首。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "我错了，我真的错了，我对不起组织和家人。",
            "那是我的责任，我一时糊涂，我愿意接受所有惩罚。",
            "不用你们问，我知道的全部如实写下来。",
            "我放松了对自己的要求，这是我罪有应得。"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "全盘交代：不再掩饰，主动、直接承认错误并补充警方尚未掌握的细节。", "max_stress": 100, "trigger_strategy": ["正常询问", "连续追问", "出示证据"]},
            {"pattern": "痛哭流涕：伴随着强烈的悔恨情绪进行陈述，表现出深深的愧疚。", "max_stress": 100, "trigger_strategy": ["心理施压", "缓和安抚"]},
            {"pattern": "自我剖析：不仅交代事实，还主动深刻反思自己是如何一步步堕落的。", "max_stress": 100, "trigger_strategy": ["缓和安抚", "正常询问"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        return "【字数建议】：控制在 300字 左右。非常愿意配合，并且夹杂了大量的悔恨心情的自我批判。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        return base_drop * 1.5 # 极易被攻破

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"缓和安抚": -2.0} # 极需心理疏导
        return base_rise * effect_map.get(strat_name, 1.0)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        # 基本全程处于配合交代的边缘
        if defense_value < 60 or stress_value > 50:
            return "full_confession"
        return "rationalization"


class IgnorantPersonality(BasePersonality):
    """懵懂无知型：对纪法认知模糊，觉得只是工作失误"""
    name = "懵懂无知型"
    initial_defense = 70.0
    initial_stress = 20.0
    volatility = 0.1

    @property
    def core_traits(self) -> str:
        return (
            "缺乏基本法律和制度认知。觉得自己的受贿、违规只是一般的“工作失误”或“行业惯例”。"
            "叙事经常夹杂无关琐碎工作流程，对于定罪有明显困惑。在被明确科普了法律后，会转变为恐慌进而交代。"
        )

    @property
    def catchphrases(self) -> list[str]:
        return [
            "这真的算违纪吗？我一直以为只是工作上的小疏忽。",
            "大家不都是这么干的吗？怎么单单就抓我违纪了？",
            "我只是按习惯办的，没想到会触犯纪律啊。",
            "如果我如实都说了，会不会因为我不清楚规定就少受点处罚？"
        ]

    @property
    def reaction_patterns(self) -> list[dict]:
        return [
            {"pattern": "天真反问：用不解的语气反问主审官这为什么属于受贿或违纪。", "max_stress": 60, "trigger_strategy": ["正常询问", "连续追问", "心理施压"]},
            {"pattern": "细节跑偏：在交代时总是把重点放在繁琐的行政流程和开会细节上，抓不住职务犯罪的重点。", "max_stress": 80, "trigger_strategy": ["正常询问", "连续追问"]},
            {"pattern": "恍然大悟：在被指出明确的法条或纪律处分条例后，表现出极其错愕和突然的恐慌。", "max_stress": 100, "trigger_strategy": ["出示证据", "心理施压"]}
        ]

    def get_verbosity_rules(self, stress_value: float) -> str:
        if stress_value < 60:
            return "【字数建议】：话中夹带非常多细枝末节的会议、流程（200字左右），让审讯官抓不到重点。"
        return "【字数建议】：发觉事情严重性后产生的惊慌失措的短回答（150字以内）。"


    def calculate_defense_damage(self, base_drop: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.5, "缓和安抚": 1.2, "心理施压": 0.8, "连续追问": 1.0}
        return base_drop * effect_map.get(strat_name, 0.9)

    def calculate_stress_increase(self, base_rise: float, strat_name: str) -> float:
        effect_map = {"出示证据": 1.6, "心理施压": 1.5, "缓和安抚": -1.5, "连续追问": 1.2}
        return base_rise * effect_map.get(strat_name, 1.0)

    def get_strategy_bias(self, defense_value: float, stress_value: float, intent: str, evidence_strength: float, is_trap: bool) -> str:
        if defense_value < 30 or stress_value > 70:
            return "full_confession"
        if evidence_strength > 0.4:
            return "rationalization" # 认为自己只是按惯例办事
        return "feign_ignorance"


# 工厂字典，保留兼容，并加上新类
PERSONALITY_FACTORY = {
    "arrogant": ArrogantPersonality, # 经典留存
    "nervous": NervousPersonality,   # 经典留存
    "cautious": CalmPersonality,     # 经典留存
    "calm": CalmPersonality,         # 经典留存
    "calculating": CalculatingPersonality,
    "anxious": AnxiousPersonality,
    "defiant": DefiantPersonality,
    "slick": SlickPersonality,
    "silent": SilentPersonality,
    "loyal": LoyalPersonality,
    "opportunistic": OpportunisticPersonality,
    "repentant": RepentantPersonality,
    "ignorant": IgnorantPersonality,
}
