#!/usr/bin/env python3
# ==================== DPSM 主测试程序 ====================
# 文件名: main_DPSM.py
# 功能: 测试动态心理状态机模块
# ======================================================

from psych_state_machine import DynamicPsychologicalStateMachine


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试1: 基本功能测试 (傲慢型人格)")
    print("=" * 60)

    # 创建心理状态机
    psych_machine = DynamicPsychologicalStateMachine("Arrogant")
    psych_machine.set_debug_mode(True)

    print("初始状态:", psych_machine.get_state_dict())
    print("人格配置:", {
        "type": psych_machine.profile.profile_type,
        "resistance": psych_machine.profile.resistance,
        "resilience": psych_machine.profile.resilience,
        "volatility": psych_machine.profile.volatility
    })

    # 模拟审讯过程
    test_scenarios = [
        {"evidence": 0.3, "trap": False, "pressure": 0.3, "style": "Normal"},
        {"evidence": 0.6, "trap": True, "pressure": 0.7, "style": "Aggressive"},
        {"evidence": 0.8, "trap": False, "pressure": 0.9, "style": "Aggressive"},
        {"evidence": 0.1, "trap": False, "pressure": 0.2, "style": "Soft"},
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'=' * 40}")
        print(f"第{i}轮审讯")
        print(f"{'=' * 40}")

        print(f"场景参数:")
        print(f"  • 证据强度: {scenario['evidence']:.1f}")
        print(f"  • 是否为陷阱: {scenario['trap']}")
        print(f"  • 压力级别: {scenario['pressure']:.1f}")
        print(f"  • 审讯风格: {scenario['style']}")

        state = psych_machine.update(
            evidence_strength=scenario["evidence"],
            is_trap=scenario["trap"],
            pressure_level=scenario["pressure"],
            interrogation_style=scenario["style"]
        )

        print(f"\n状态更新结果:")
        print(f"  • 防御值: {state['defense_value']:.1f}")
        print(f"  • 压力值: {state['stress_value']:.1f}")
        print(f"  • 状态标签: {state['status_label']}")
        print(f"  • 崩溃风险: {psych_machine.get_risk_level()}")
        print(f"  • 防御趋势: {psych_machine.get_defense_trend():.2f}")
        print(f"  • 压力趋势: {psych_machine.get_stress_trend():.2f}")
        print(f"  • 建议策略: {psych_machine.get_suggested_approach()['approach']}")

    # 显示历史摘要
    print(f"\n{'=' * 60}")
    print("审讯历史摘要:")
    print('=' * 60)
    summary = psych_machine.get_history_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")


def test_different_personalities():
    """测试不同人格类型"""
    print(f"\n{'=' * 60}")
    print("测试2: 不同人格类型对比")
    print('=' * 60)

    personalities = ["Arrogant", "Calm", "Nervous"]

    for personality in personalities:
        print(f"\n{'-' * 40}")
        print(f"测试 {personality} 型人格")
        print(f"{'-' * 40}")

        machine = DynamicPsychologicalStateMachine(personality)

        # 应用相同的审讯场景
        scenarios = [
            {"evidence": 0.5, "trap": False, "pressure": 0.5, "style": "Normal"},
            {"evidence": 0.7, "trap": False, "pressure": 0.8, "style": "Aggressive"},
        ]

        for i, scenario in enumerate(scenarios, 1):
            machine.update(
                evidence_strength=scenario["evidence"],
                is_trap=scenario["trap"],
                pressure_level=scenario["pressure"],
                interrogation_style=scenario["style"]
            )

        state = machine.get_state_dict()
        print(f"最终状态:")
        print(f"  • 防御值: {state['defense_value']:.1f}")
        print(f"  • 压力值: {state['stress_value']:.1f}")
        print(f"  • 状态标签: {state['status_label']}")
        print(f"  • 崩溃风险: {machine.get_risk_level()}")


def test_reset_functionality():
    """测试重置功能"""
    print(f"\n{'=' * 60}")
    print("测试3: 重置功能测试")
    print('=' * 60)

    machine = DynamicPsychologicalStateMachine("Arrogant")

    # 进行几轮审讯
    for _ in range(3):
        machine.update(
            evidence_strength=0.6,
            is_trap=False,
            pressure_level=0.7,
            interrogation_style="Aggressive"
        )

    print("重置前状态:")
    state_before = machine.get_state_dict()
    print(f"  • 轮数: {state_before['round']}")
    print(f"  • 防御值: {state_before['defense_value']:.1f}")
    print(f"  • 压力值: {state_before['stress_value']:.1f}")

    # 重置为紧张型人格
    machine.reset("Nervous")

    print("\n重置后状态:")
    state_after = machine.get_state_dict()
    print(f"  • 轮数: {state_after['round']}")
    print(f"  • 防御值: {state_after['defense_value']:.1f}")
    print(f"  • 压力值: {state_after['stress_value']:.1f}")
    print(f"  • 人格类型: {state_after['profile_type']}")


def test_edge_cases():
    """测试边界情况"""
    print(f"\n{'=' * 60}")
    print("测试4: 边界情况测试")
    print('=' * 60)

    machine = DynamicPsychologicalStateMachine("Nervous")

    # 测试极强证据
    print("\n1. 极强证据测试:")
    machine.update(
        evidence_strength=1.0,
        is_trap=False,
        pressure_level=1.0,
        interrogation_style="Aggressive"
    )
    print(f"  防御值: {machine.get_state_dict()['defense_value']:.1f}")
    print(f"  是否崩溃: {machine.is_broken()}")

    # 重置
    machine.reset()

    # 测试多次识破陷阱
    print("\n2. 多次识破陷阱测试:")
    for _ in range(3):
        machine.update(
            evidence_strength=0.3,
            is_trap=True,
            pressure_level=0.3,
            interrogation_style="Normal"
        )
    print(f"  防御值: {machine.get_state_dict()['defense_value']:.1f}")
    print(f"  状态标签: {machine.get_state_dict()['status_label']}")


def interactive_test():
    """交互式测试"""
    print(f"\n{'=' * 60}")
    print("交互式测试模式")
    print('=' * 60)

    personality = input("选择人格类型 (Arrogant/Calm/Nervous, 默认Arrogant): ") or "Arrogant"

    machine = DynamicPsychologicalStateMachine(personality)

    while True:
        print(f"\n当前状态:")
        state = machine.get_state_dict()
        print(f"  轮数: {state['round']}")
        print(f"  防御值: {state['defense_value']:.1f}")
        print(f"  压力值: {state['stress_value']:.1f}")
        print(f"  状态: {state['status_label']}")
        print(f"  风险等级: {machine.get_risk_level()}")

        if machine.is_broken():
            print("⚠️  心理防线已崩溃！")
            break

        print("\n输入审讯参数:")
        try:
            evidence = float(input("  证据强度 (0.0-1.0): "))
            trap_input = input("  是否为陷阱 (y/n): ").lower() == 'y'
            pressure = float(input("  压力级别 (0.0-1.0): "))
            style = input("  审讯风格 (Normal/Aggressive/Soft, 默认Normal): ") or "Normal"

            machine.update(
                evidence_strength=evidence,
                is_trap=trap_input,
                pressure_level=pressure,
                interrogation_style=style
            )

            suggestion = machine.get_suggested_approach()
            print(f"\n建议策略: {suggestion['approach']}")
            print(f"策略说明: {suggestion['reason']}")

        except ValueError:
            print("输入错误，请重新输入")
            continue

        continue_test = input("\n继续测试? (y/n): ").lower()
        if continue_test != 'y':
            break

    # 显示最终摘要
    print(f"\n{'=' * 40}")
    print("最终审讯摘要")
    print('=' * 40)
    summary = machine.get_history_summary()
    print(f"总轮数: {summary['total_rounds']}")
    print(f"平均防御值: {summary['average_defense']:.1f}")
    print(f"平均压力值: {summary['average_stress']:.1f}")
    print(f"最终状态: {summary['current_status']}")
    print(f"是否崩溃: {summary['broken_status']}")


def main():
    """主函数"""
    print("动态心理状态机 (DPSM) 测试程序")
    print("版本: 2.0")
    print("=" * 60)

    # 运行所有测试
    test_basic_functionality()
    test_different_personalities()
    test_reset_functionality()
    test_edge_cases()

    # 可选: 运行交互式测试
    run_interactive = input("\n是否运行交互式测试? (y/n): ").lower()
    if run_interactive == 'y':
        interactive_test()

    print(f"\n{'=' * 60}")
    print("所有测试完成!")
    print('=' * 60)


if __name__ == "__main__":
    main()