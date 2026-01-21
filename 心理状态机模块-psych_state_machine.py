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

    # 人格特定的动力学参数
    ALPHA_PARAMS = {  # 证据伤害系数
        "Arrogant": 15.0,
        "Calm": 12.0,
        "Nervous": 18.0
    }

    BETA_PARAMS = {  # 压力传导系数
        "Arrogant": 8.5,
        "Calm": 6.0,
        "Nervous": 10.0
    }

    GAMMA_PARAMS = {  # 自然衰减率
        "Arrogant": 8.9,
        "Calm": 10.0,
        "Nervous": 7.5
    }

    # 恢复系数
    RECOVERY_PARAMS = {
        "Arrogant": 0.7,  # 傲慢型恢复较慢
        "Calm": 1.0,  # 冷静型恢复正常
        "Nervous": 0.5  # 紧张型恢复较慢
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

        # 动力学参数（根据人格类型调整）
        self.alpha = self.profile.ALPHA_PARAMS.get(profile_type, 15.0)  # 证据伤害系数
        self.beta = self.profile.BETA_PARAMS.get(profile_type, 8.5)  # 压力传导系数
        self.gamma = self.profile.GAMMA_PARAMS.get(profile_type, 8.9)  # 自然衰减率
        self.recovery_coeff = self.profile.RECOVERY_PARAMS.get(profile_type, 1.0)  # 恢复系数

        # 历史记录
        self.history: list = []
        self.round_count = 0

        # 状态标签
        self._update_status_label()

        # 调试模式
        self.debug_mode = False

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

    def _log(self, message: str):
        """调试日志"""
        if self.debug_mode:
            print(f"[DPSM调试] {message}")

    def set_debug_mode(self, enabled: bool):
        """设置调试模式"""
        self.debug_mode = enabled

    def update(self,
               evidence_strength: float,
               is_trap: bool,
               pressure_level: float = 0.5,
               interrogation_style: str = "Normal") -> Dict:
        """
        更新心理状态

        Args:
            evidence_strength: 证据强度 (0.0-1.0)
            is_trap: 是否为陷阱问题
            pressure_level: 压力级别 (0.0-1.0)
            interrogation_style: 审讯风格 (Normal/Aggressive/Soft)

        Returns:
            更新后的心理状态字典
        """
        self.round_count += 1

        self._log(f"第{self.round_count}轮更新:")
        self._log(
            f"  输入: 证据强度={evidence_strength:.2f}, 陷阱={is_trap}, 压力={pressure_level}, 风格={interrogation_style}")

        # 1. 审讯风格调整系数
        style_multiplier = {
            "Normal": 1.0,
            "Aggressive": 1.3,  # 激进审讯增加压力
            "Soft": 0.7  # 温和审讯减少压力
        }.get(interrogation_style, 1.0)

        # 2. 压力更新 (Stress Dynamics)
        # 压力来源: 证据冲击 + 审讯压力 + 陷阱压力
        stress_increment = (evidence_strength * self.alpha) + (pressure_level * self.beta * style_multiplier)

        if is_trap:
            # 识破陷阱: 压力骤降，自信恢复
            stress_increment -= 15.0 * self.recovery_coeff
            self._log("  识破陷阱！压力降低，信心恢复")
        elif evidence_strength > 0.8:
            # 极强证据：额外压力
            stress_increment += 5.0
            self._log("  极强证据！额外压力增加")

        self.stress = self.stress - self.gamma + stress_increment
        self.stress = max(0.0, min(100.0, self.stress))

        # 3. 防御更新 (Defense Dynamics)
        # 基础伤害 = 证据强度 × 伤害系数 × 人格抗压系数
        base_damage = evidence_strength * self.alpha * (1 - self.profile.resistance)

        # 考虑情绪波动性
        if self.profile.volatility > 0.5:
            base_damage *= 1.2  # 高波动性增加伤害

        # 高压倍增器: 压力>60时伤害增加
        stress_multiplier = 1.5 if self.stress > 60 else 1.0

        # 审讯风格对伤害的影响
        if interrogation_style == "Aggressive" and evidence_strength > 0.4:
            stress_multiplier *= 1.2  # 激进审讯对中等以上证据增强伤害

        total_damage = base_damage * stress_multiplier

        # 恢复计算
        recovery = 0.0

        # 人格恢复能力
        base_recovery = self.profile.resilience * 10.0

        if is_trap:
            recovery += 20.0 * self.recovery_coeff  # 识破陷阱，信心大增
        elif evidence_strength < 0.2:
            recovery += 8.0 * self.recovery_coeff  # 证据不足，稍作恢复
        elif pressure_level < 0.3:
            recovery += 5.0 * self.recovery_coeff  # 压力较小，略有恢复

        # 自然恢复（防御值越低，恢复越困难）
        if self.defense > 30:
            recovery += base_recovery
        elif self.defense > 10:
            recovery += base_recovery * 0.5

        # 应用更新
        self.defense = self.defense - total_damage + recovery
        self.defense = max(0.0, min(100.0, self.defense))

        self._log(f"  伤害计算: 基础={base_damage:.1f} × 压力倍数={stress_multiplier:.1f} = 总伤害={total_damage:.1f}")
        self._log(f"  恢复: {recovery:.1f} (基础恢复={base_recovery:.1f})")
        self._log(f"  防御: {self.defense:.1f} | 压力: {self.stress:.1f}")

        # 4. 更新状态标签
        self._update_status_label()

        # 5. 记录历史
        state_record = {
            "round": self.round_count,
            "defense": self.defense,
            "stress": self.stress,
            "evidence_strength": evidence_strength,
            "is_trap": is_trap,
            "pressure_level": pressure_level,
            "interrogation_style": interrogation_style,
            "status_label": self.status_label.value,
            "damage_taken": total_damage,
            "recovery_gained": recovery,
            "timestamp": time.time()
        }

        self.history.append(state_record)

        return self.get_state_dict()

    def get_state_dict(self) -> Dict:
        """获取当前心理状态字典"""
        return {
            "defense_value": round(self.defense, 2),
            "stress_value": round(self.stress, 2),
            "status_label": self.status_label.value,
            "profile_type": self.profile.profile_type,
            "round": self.round_count,
            "resistance": self.profile.resistance,
            "resilience": self.profile.resilience,
            "volatility": self.profile.volatility
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

    def get_stress_trend(self, window: int = 5) -> float:
        """
        获取压力值变化趋势

        Args:
            window: 计算趋势的窗口大小

        Returns:
            压力值变化率 (正数表示上升，负数表示下降)
        """
        if len(self.history) < 2:
            return 0.0

        recent_history = self.history[-min(window, len(self.history)):]

        if len(recent_history) < 2:
            return 0.0

        total_change = 0.0
        for i in range(1, len(recent_history)):
            change = recent_history[i]["stress"] - recent_history[i - 1]["stress"]
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

    def get_suggested_approach(self) -> Dict:
        """
        获取建议的审讯策略
        基于当前心理状态提供策略建议
        """
        if self.is_broken():
            return {
                "approach": "Direct Confrontation",
                "intensity": "Low",
                "reason": "心理防线已崩溃，适合直接质问获取供词"
            }
        elif self.status_label == PsychologicalStateLabel.ANXIOUS:
            return {
                "approach": "Pressure",
                "intensity": "High",
                "reason": "焦虑状态下，持续施压可能突破防线"
            }
        elif self.status_label == PsychologicalStateLabel.HESITANT:
            return {
                "approach": "Good Cop/Bad Cop",
                "intensity": "Medium",
                "reason": "犹豫不决时，红白脸策略有效"
            }
        elif self.status_label == PsychologicalStateLabel.DEFENSIVE:
            return {
                "approach": "Evidence Presentation",
                "intensity": "Medium-High",
                "reason": "防御状态下，展示证据逐步施压"
            }
        else:  # CALM
            return {
                "approach": "Long Game",
                "intensity": "Low-Medium",
                "reason": "冷静状态下，需要耐心和策略"
            }

    def get_history_summary(self) -> Dict:
        """获取审讯历史摘要"""
        if not self.history:
            return {"total_rounds": 0, "average_defense": 0, "average_stress": 0}

        avg_defense = sum(h["defense"] for h in self.history) / len(self.history)
        avg_stress = sum(h["stress"] for h in self.history) / len(self.history)

        # 统计状态分布
        state_counts = {}
        for h in self.history:
            state = h["status_label"]
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "total_rounds": len(self.history),
            "average_defense": round(avg_defense, 2),
            "average_stress": round(avg_stress, 2),
            "state_distribution": state_counts,
            "current_status": self.status_label.value,
            "broken_status": self.is_broken()
        }

    def reset(self, new_profile_type: Optional[str] = None):
        """重置心理状态，可选择新的人格类型"""
        if new_profile_type:
            self.profile = PsychologicalProfile(new_profile_type)

        profile_type = self.profile.profile_type
        self.defense = self.profile.INITIAL_DEFENSE.get(profile_type, 85.0)
        self.stress = self.profile.INITIAL_STRESS.get(profile_type, 15.0)

        # 重新初始化动力学参数
        self.alpha = self.profile.ALPHA_PARAMS.get(profile_type, 15.0)
        self.beta = self.profile.BETA_PARAMS.get(profile_type, 8.5)
        self.gamma = self.profile.GAMMA_PARAMS.get(profile_type, 8.9)
        self.recovery_coeff = self.profile.RECOVERY_PARAMS.get(profile_type, 1.0)

        self.history = []
        self.round_count = 0
        self._update_status_label()

        self._log(f"已重置，人格类型: {profile_type}")
