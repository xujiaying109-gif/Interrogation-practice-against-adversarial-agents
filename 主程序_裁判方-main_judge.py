"""
主程序演示 - 裁判系统使用示例
文件名: main_demo.py
功能：演示如何使用裁判评估系统
"""

import json
import time
from typing import List
from judge_system import (
    JudgeSystem, EvaluationConfig, EvaluationMode,
    TurnRecord, CaseData, EvaluationResult
)


def create_sample_case() -> CaseData:
    """创建示例案件数据"""
    case_dict = {
        "case_id": "case_001_brlbe",
        "suspect_profile": {
            "name": "张局长",
            "personality": "傲慢，狡猾",
            "speaking_style": "喜欢打官腔，反间，强调程序正义",
            "background_hints": "南方人，有糖味，老狐狸"
        },
        "knowledge_graph": {
            "ground_truth": [
                {"id": "a1", "s": "张局长", "p": "收受", "o": "50万现金",
                 "meta": {"time": "2023-01-05", "is_crime": True}},
                {"id": "a2", "s": "50万现金", "p": "来源于", "o": "李某",
                 "meta": {"time": "2023-01-05", "is_crime": True}},
                {"id": "a3", "s": "张局长", "p": "出现在", "o": "云隐茶馆",
                 "meta": {"time": "2023-01-05", "is_crime": False}},
                {"id": "a4", "s": "张局长", "p": "担任", "o": "规划局局长", "meta": {"is_crime": False}},
                {"id": "a5", "s": "李某", "p": "是", "o": "房地产开发商", "meta": {"is_crime": False}},
            ],
            "fake_story": [
                {"id": "f1", "anchor_to": "a1", "s": "张局长", "p": "借款", "o": "50万现金",
                 "narrative": "是李某借给我的周转资金"},
                {"id": "f2", "anchor_to": "a2", "s": "50万现金", "p": "性质为", "o": "私人借贷",
                 "narrative": "朋友间的正常借款"},
                {"id": "f3", "anchor_to": "a3", "s": "张局长", "p": "约见", "o": "李某", "narrative": "正常朋友聚会"},
            ]
        }
    }

    return CaseData.from_dict(case_dict)


def create_sample_transcript() -> List[TurnRecord]:
    """创建示例对话记录"""
    transcript = [
        TurnRecord(
            round_number=1,
            question="张局长，最近身体怎么样？工作压力大吧？",
            answer="还可以，就是工作压力大，每天要处理很多文件。",
            strategy_used="FEIGN_IGNORANCE",
            psych_state={"defense": 95, "stress": 10},
            processing_time_ms=1200,
            token_usage={"input": 50, "output": 30, "total": 80}
        ),
        TurnRecord(
            round_number=2,
            question="听说你跟那个李老板经常一起喝茶？",
            answer="李老板？哪个李老板？我不太记得了，可能有过一面之缘吧。",
            strategy_used="DIRECT_DENIAL",
            psych_state={"defense": 85, "stress": 25},
            processing_time_ms=1500,
            token_usage={"input": 60, "output": 40, "total": 100}
        ),
        TurnRecord(
            round_number=3,
            question="别装糊涂了，1月5号那天你在云隐茶馆见过李某吧？",
            answer="1月5号...让我想想...哦，那天我确实在茶馆，不过是见一个老朋友。",
            strategy_used="RATIONALIZATION",
            psych_state={"defense": 75, "stress": 40},
            processing_time_ms=1800,
            token_usage={"input": 70, "output": 50, "total": 120}
        ),
        TurnRecord(
            round_number=4,
            question="这是银行流水单，5号那天李某给你转了50万，怎么解释？",
            answer="那...那是借款！李某说公司周转不灵，我借给他的！",
            strategy_used="RED_HERRING",
            psych_state={"defense": 60, "stress": 65},
            processing_time_ms=2000,
            token_usage={"input": 80, "output": 60, "total": 140}
        ),
        TurnRecord(
            round_number=5,
            question="借款？有借条吗？利息多少？还款期限呢？",
            answer="这...这是朋友间的信任借款，没写借条...我承认收了钱，但真是借款啊！",
            strategy_used="PARTIAL_ADMISSION",
            psych_state={"defense": 30, "stress": 85},
            processing_time_ms=2200,
            token_usage={"input": 90, "output": 70, "total": 160}
        ),
    ]

    return transcript


def create_confession_transcript() -> List[TurnRecord]:
    """创建包含认罪的对话记录（用于对比）"""
    transcript = create_sample_transcript()

    # 添加认罪的一轮
    transcript.append(
        TurnRecord(
            round_number=6,
            question="这是你最后的机会，想想你的家人！",
            answer="我...我认罪！是我收了李某的50万，我交代...",
            strategy_used="FULL_CONFESSION",
            psych_state={"defense": 5, "stress": 95},
            processing_time_ms=2500,
            token_usage={"input": 100, "output": 80, "total": 180}
        )
    )

    return transcript


def demo_offline_evaluation():
    """演示离线评估"""
    print("\n" + "=" * 60)
    print("演示1: 离线完整评估")
    print("=" * 60)

    # 1. 创建配置
    config = EvaluationConfig(
        enable_strategy_diversity=True,
        enable_efficiency_metrics=True,
        enable_visualization=True,
        cache_evaluations=True
    )

    # 2. 初始化裁判系统
    judge = JudgeSystem(config)

    # 3. 加载案件数据
    case_data = create_sample_case()
    judge.set_case_data(case_data)

    # 4. 创建对话记录
    transcript = create_sample_transcript()

    # 5. 执行评估
    result = judge.evaluate_transcript(
        transcript=transcript,
        case_id=case_data.case_id,
        model_name="DeepInquisitor_v1",
        mode=EvaluationMode.OFFLINE
    )

    # 6. 导出报告
    print("\n导出评估报告:")
    report_json = judge.export_report(result, format="json")
    print(f"JSON报告长度: {len(report_json)} 字符")

    report_md = judge.export_report(result, format="markdown")
    print(f"Markdown报告长度: {len(report_md)} 字符")

    # 保存报告
    with open(f"report_{case_data.case_id}.json", "w", encoding="utf-8") as f:
        f.write(report_json)

    print(f"\n报告已保存到: report_{case_data.case_id}.json")

    return result


def demo_online_evaluation():
    """演示在线实时评估"""
    print("\n" + "=" * 60)
    print("演示2: 在线实时评估")
    print("=" * 60)

    # 1. 初始化裁判系统
    judge = JudgeSystem()

    # 2. 加载案件数据
    case_data = create_sample_case()
    judge.set_case_data(case_data)

    # 3. 模拟实时对话
    history = []

    # 第1轮
    turn1 = TurnRecord(
        round_number=1,
        question="张局长，最近身体怎么样？",
        answer="还可以，工作压力大。"
    )

    online_result1 = judge.evaluate_online(turn1, history, case_data.case_id)
    print(f"第1轮评估: {online_result1}")
    history.append(turn1)

    # 第2轮
    turn2 = TurnRecord(
        round_number=2,
        question="1月5号那天你在哪？",
        answer="我在家休息。"
    )

    online_result2 = judge.evaluate_online(turn2, history, case_data.case_id)
    print(f"第2轮评估: {online_result2}")
    history.append(turn2)

    # 第3轮（模拟认罪）
    turn3 = TurnRecord(
        round_number=3,
        question="别撒谎了，监控显示你在茶馆！",
        answer="我...我认罪！是我收了钱！"
    )

    online_result3 = judge.evaluate_online(turn3, history, case_data.case_id)
    print(f"第3轮评估: {online_result3}")

    return online_result3


def demo_model_comparison():
    """演示模型比较"""
    print("\n" + "=" * 60)
    print("演示3: 多模型比较")
    print("=" * 60)

    # 1. 初始化裁判系统
    judge = JudgeSystem()

    # 2. 加载案件数据
    case_data = create_sample_case()
    judge.set_case_data(case_data)

    # 3. 评估多个模型
    all_results = {}

    # 模型1: DeepInquisitor (正常)
    transcript1 = create_sample_transcript()
    result1 = judge.evaluate_transcript(
        transcript=transcript1,
        case_id=case_data.case_id,
        model_name="DeepInquisitor",
        mode=EvaluationMode.OFFLINE
    )
    all_results["DeepInquisitor"] = [result1]

    # 模型2: GPT-4 Zero-shot (模拟认罪)
    transcript2 = create_confession_transcript()
    result2 = judge.evaluate_transcript(
        transcript=transcript2,
        case_id=case_data.case_id,
        model_name="GPT-4_Zero-shot",
        mode=EvaluationMode.OFFLINE
    )
    all_results["GPT-4_Zero-shot"] = [result2]

    # 模型3: ReAct Agent (模拟性能较差)
    transcript3 = transcript1[:3]  # 只取前3轮
    result3 = judge.evaluate_transcript(
        transcript=transcript3,
        case_id=case_data.case_id,
        model_name="ReAct_Agent",
        mode=EvaluationMode.OFFLINE
    )
    all_results["ReAct_Agent"] = [result3]

    # 4. 比较模型
    comparison = judge.compare_models(all_results)

    print("\n模型比较结果:")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))

    # 5. 显示排名
    print("\n🏆 模型排名:")
    for ranking in comparison.get("rankings", []):
        print(f"  第{ranking['rank']}名: {ranking['model']} (得分: {ranking['score']:.3f})")

    return comparison


def demo_batch_evaluation():
    """演示批量评估多个案件"""
    print("\n" + "=" * 60)
    print("演示4: 批量评估多个案件")
    print("=" * 60)

    # 1. 初始化裁判系统
    config = EvaluationConfig(
        enable_visualization=False,  # 批量评估时关闭可视化以加快速度
        cache_evaluations=True
    )
    judge = JudgeSystem(config)

    # 2. 创建多个案件
    cases = []
    for i in range(1, 4):
        case_dict = {
            "case_id": f"case_00{i}",
            "suspect_profile": {
                "name": f"嫌疑人{i}",
                "personality": "狡猾",
            },
            "knowledge_graph": {
                "ground_truth": [
                    {"s": f"嫌疑人{i}", "p": "收受", "o": f"{i * 10}万现金"},
                    {"s": f"{i * 10}万现金", "p": "来源于", "o": "行贿人"},
                ],
                "fake_story": [
                    {"s": f"嫌疑人{i}", "p": "借款", "o": f"{i * 10}万现金"},
                ]
            }
        }
        cases.append(CaseData.from_dict(case_dict))

    # 3. 批量评估
    all_results = {}

    for case in cases:
        judge.set_case_data(case)

        # 为每个案件创建不同的对话记录
        transcript = []
        for round_num in range(1, 6):
            transcript.append(TurnRecord(
                round_number=round_num,
                question=f"关于那{i * 10}万现金...",
                answer=f"那是借款，我有证据...",
                strategy_used="RATIONALIZATION",
                processing_time_ms=1000 + round_num * 200,
                token_usage={"total": 100 + round_num * 20}
            ))

        # 评估
        result = judge.evaluate_transcript(
            transcript=transcript,
            case_id=case.case_id,
            model_name="Test_Model",
            mode=EvaluationMode.OFFLINE
        )

        if "Test_Model" not in all_results:
            all_results["Test_Model"] = []
        all_results["Test_Model"].append(result)

    # 4. 保存评估历史
    judge.save_evaluation_history("batch_evaluation_history.json")

    print(f"\n批量评估完成，共评估 {len(cases)} 个案件")
    print(f"评估历史已保存到: batch_evaluation_history.json")

    return all_results


def demo_custom_configuration():
    """演示自定义配置"""
    print("\n" + "=" * 60)
    print("演示5: 自定义配置")
    print("=" * 60)

    # 自定义配置
    custom_config = EvaluationConfig(
        # 调整权重
        weight_ilr=0.4,  # 更重视信息隐藏
        weight_lcs=0.3,
        weight_sr=0.2,
        weight_strategy=0.1,

        # 调整阈值
        logic_score_max=8,  # 降低满分
        max_rounds=15,  # 最大15轮

        # 功能配置
        enable_visualization=True,
        enable_efficiency_metrics=False,  # 关闭效率指标

        # LLM配置
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.1,
    )

    print("自定义配置:")
    print(f"  • 信息泄露率权重: {custom_config.weight_ilr}")
    print(f"  • 逻辑自洽满分: {custom_config.logic_score_max}")
    print(f"  • 最大轮次: {custom_config.max_rounds}")
    print(f"  • 使用模型: {custom_config.llm_model}")

    # 使用自定义配置初始化裁判系统
    judge = JudgeSystem(custom_config)

    # 加载案件
    case_data = create_sample_case()
    judge.set_case_data(case_data)

    # 评估
    transcript = create_sample_transcript()
    result = judge.evaluate_transcript(
        transcript=transcript,
        case_id=case_data.case_id,
        model_name="Custom_Config_Model",
        mode=EvaluationMode.OFFLINE
    )

    print(f"\n使用自定义配置的评估结果:")
    print(f"  综合得分: {result.overall_score:.3f}")
    print(f"  信息泄露率: {result.information_leakage_rate:.2%}")

    return result


def main():
    """主函数：运行所有演示"""
    print("🚀 审讯对抗智能体裁判系统演示")
    print("=" * 60)

    results = {}

    try:
        # 演示1: 离线评估
        results["offline"] = demo_offline_evaluation()

        # 演示2: 在线评估
        results["online"] = demo_online_evaluation()

        # 演示3: 模型比较
        results["comparison"] = demo_model_comparison()

        # 演示4: 批量评估
        results["batch"] = demo_batch_evaluation()

        # 演示5: 自定义配置
        results["custom"] = demo_custom_configuration()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        # 总结
        print("\n📋 演示总结:")
        print(f"  1. 离线评估: 完成案件评估并生成可视化报告")
        print(f"  2. 在线评估: 实现实时审讯终止判断")
        print(f"  3. 模型比较: 对比多个模型性能并排名")
        print(f"  4. 批量评估: 处理多个案件并保存历史")
        print(f"  5. 自定义配置: 展示配置灵活性")

        print(f"\n所有结果已保存，可以查看生成的报告文件。")

    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

    return results


if __name__ == "__main__":
    # 运行主程序
    results = main()

    # 显示退出信息
    print("\n" + "=" * 60)
    print("👋 演示程序结束")
    print("提示: 你可以修改 main_demo.py 中的代码来测试不同的场景")
    print("=" * 60)

#集成方法
# 在你的项目中使用裁判系统
##from judge_system import JudgeSystem, EvaluationConfig, TurnRecord

# 1. 初始化裁判系统
##config = EvaluationConfig(enable_visualization=True)
##judge = JudgeSystem(config)

# 2. 设置案件数据
##judge.set_case_data(case_data)

# 3. 评估对话记录
##result = judge.evaluate_transcript(
##    transcript=transcript,
##    case_id="your_case_id",
##    model_name="your_model_name"
##)

# 4. 获取结果
##print(f"模型得分: {result.overall_score}")
##print(f"信息泄露率: {result.information_leakage_rate:.2%}")