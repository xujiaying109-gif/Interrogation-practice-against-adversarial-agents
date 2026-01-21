# ==================== 动态心理状态机 (DPSM) ====================
import time
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PsychologicalStateLabel(str, Enum):
    """心理状态标签"""
    CALM = "CALM"  # 冷静：防御>80
    DEFENSIVE = "DEFENSIVE"  # 防御：防御50-80
    HESITANT = "HESITANT"  # 犹豫：防御<50且压力<60
    ANXIOUS = "ANXIOUS"  # 焦虑：防御<50且压力>60
    BROKEN = "BROKEN"  # 崩溃：防御<20


@dataclass
class PsychologicalProfile:
    """心理人格配置"""
    profile_type: str = "Arrogant"  # 人格类型：Arrogant/Calm/Nervous
    resistance: float = 0.8  # 抗压能力 (0-1, 越高越难攻破)
    resilience: float = 0.3  # 恢复能力 (0-1, 越高恢复越快)
    volatility: float = 0.2  # 情绪波动性 (0-1, 越高对证据越敏感)
    # 不同人格的初始值
    INITIAL_DEFENSE = {
        "Arrogant": 98.0,  # 傲慢型：初始防御高
        "Calm": 85.0,  # 冷静型：中等防御
        "Nervous": 70.0  # 紧张型：初始防御低
    }
    INITIAL_STRESS = {
        "Arrogant": 10.0,  # 傲慢型：初始压力低
        "Calm": 15.0,  # 冷静型：中等压力
        "Nervous": 25.0  # 紧张型：初始压力高
    }


class DynamicPsychologicalStateMachine:
    """
    动态心理状态机 (DPSM)
    模拟嫌疑人在审讯过程中的心理状态变化
    基于论文中的动力学方程：S(t+1) = A*S(t) + B*U(t)
    """

    def __init__(self, profile_type: str = "Arrogant"):
        # 人格配置
        self.profile = PsychologicalProfile(profile_type)

        # 状态变量
        self.defense = self.profile.INITIAL_DEFENSE.get(profile_type, 85.0)
        self.stress = self.profile.INITIAL_STRESS.get(profile_type, 15.0)

        # 动力学参数
        self.alpha = 15.0  # 证据伤害系数
        self.beta = 8.5  # 压力传导系数
        self.gamma = 8.9  # 自然衰减率

        # 历史记录
        self.history: list = []
        self.round_count = 0

        # 状态标签
        self._update_status_label()

    def _update_status_label(self) -> None:
        """根据防御值和压力值更新状态标签"""
        if self.defense <= 20:
            self.status_label = PsychologicalStateLabel.BROKEN
        elif self.defense <= 50:
            if self.stress > 60:
                self.status_label = PsychologicalStateLabel.ANXIOUS
            else:
                self.status_label = PsychologicalStateLabel.HESITANT
        elif self.defense > 80:
            self.status_label = PsychologicalStateLabel.CALM
        else:
            self.status_label = PsychologicalStateLabel.DEFENSIVE

    def update(self,
               evidence_strength: float,
               is_trap: bool,
               pressure_level: float = 0.5) -> Dict:
        """
        更新心理状态

        Args:
            evidence_strength: 证据强度 (0.0-1.0)
            is_trap: 是否为陷阱问题
            pressure_level: 压力级别 (0.0-1.0)

        Returns:
            更新后的心理状态字典
        """
        self.round_count += 1

        print(f"\n[心理状态机] 第{self.round_count}轮更新:")
        print(f"  输入: 证据强度={evidence_strength:.2f}, 陷阱={is_trap}, 压力={pressure_level}")

        # 1. 压力更新 (Stress Dynamics)
        # 压力来源: 证据冲击 + 审讯压力 + 陷阱压力
        stress_increment = (evidence_strength * self.alpha) + (pressure_level * self.beta)

        if is_trap:
            # 识破陷阱: 压力骤降，自信恢复
            stress_increment -= 10.0
            print("  识破陷阱！压力降低，信心恢复")

        self.stress = self.stress - self.gamma + stress_increment
        self.stress = max(0.0, min(100.0, self.stress))

        # 2. 防御更新 (Defense Dynamics)
        # 基础伤害 = 证据强度 × 伤害系数
        base_damage = evidence_strength * self.alpha

        # 高压倍增器: 压力>60时伤害增加
        stress_multiplier = 1.5 if self.stress > 60 else 1.0
        total_damage = base_damage * stress_multiplier

        # 恢复计算
        recovery = 0.0
        if is_trap:
            recovery += 15.0  # 识破陷阱，信心大增
        elif evidence_strength < 0.2:
            recovery += 5.0  # 证据不足，稍作恢复
        elif pressure_level < 0.3:
            recovery += 3.0  # 压力较小，略有恢复

        # 应用更新
        self.defense = self.defense - total_damage + recovery
        self.defense = max(0.0, min(100.0, self.defense))

        print(f"  伤害计算: 基础={base_damage:.1f} × 压力倍数={stress_multiplier} = 总伤害={total_damage:.1f}")
        print(f"  恢复: {recovery:.1f}")
        print(f"  防御: {self.defense:.1f} | 压力: {self.stress:.1f}")

        # 3. 更新状态标签
        self._update_status_label()

        # 4. 记录历史
        self.history.append({
            "round": self.round_count,
            "defense": self.defense,
            "stress": self.stress,
            "evidence_strength": evidence_strength,
            "is_trap": is_trap,
            "pressure_level": pressure_level,
            "status_label": self.status_label.value,
            "timestamp": time.time()
        })

        return self.get_state_dict()

    def get_state_dict(self) -> Dict:
        """获取当前心理状态字典"""
        return {
            "defense_value": round(self.defense, 2),
            "stress_value": round(self.stress, 2),
            "status_label": self.status_label.value,
            "profile_type": self.profile.profile_type,
            "round": self.round_count
        }

    def get_defense_trend(self, window: int = 5) -> float:
        """
        获取防御值变化趋势

        Args:
            window: 计算趋势的窗口大小

        Returns:
            防御值变化率 (正数表示上升，负数表示下降)
        """
        if len(self.history) < 2:
            return 0.0

        # 取最近window个数据点
        recent_history = self.history[-min(window, len(self.history)):]

        if len(recent_history) < 2:
            return 0.0

        # 计算平均变化率
        total_change = 0.0
        for i in range(1, len(recent_history)):
            change = recent_history[i]["defense"] - recent_history[i - 1]["defense"]
            total_change += change

        avg_change = total_change / (len(recent_history) - 1)
        return avg_change

    def is_broken(self) -> bool:
        """判断是否心理防线已崩溃"""
        return self.status_label == PsychologicalStateLabel.BROKEN

    def get_risk_level(self) -> str:
        """获取崩溃风险等级"""
        if self.defense <= 20:
            return "CRITICAL"
        elif self.defense <= 40:
            return "HIGH"
        elif self.defense <= 60:
            return "MEDIUM"
        else:
            return "LOW"

    def reset(self):
        """重置心理状态"""
        profile_type = self.profile.profile_type
        self.defense = self.profile.INITIAL_DEFENSE.get(profile_type, 85.0)
        self.stress = self.profile.INITIAL_STRESS.get(profile_type, 15.0)
        self.history = []
        self.round_count = 0
        self._update_status_label()
        print("[心理状态机] 已重置")

# 使用示例
if __name__ == "__main__":
    # 创建心理状态机（傲慢型人格）
    psych_machine = DynamicPsychologicalStateMachine("Arrogant")

    print("初始状态:", psych_machine.get_state_dict())

    # 模拟审讯过程
    test_scenarios = [
        {"evidence": 0.3, "trap": False, "pressure": 0.3},  # 轻度证据
        {"evidence": 0.6, "trap": True, "pressure": 0.7},  # 强证据+陷阱
        {"evidence": 0.8, "trap": False, "pressure": 0.9},  # 极强证据
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n=== 第{i}轮审讯 ===")
        state = psych_machine.update(
            evidence_strength=scenario["evidence"],
            is_trap=scenario["trap"],
            pressure_level=scenario["pressure"]
        )
        print("当前状态:", state)
        print("崩溃风险:", psych_machine.get_risk_level())
        print("防御趋势:", psych_machine.get_defense_trend())
