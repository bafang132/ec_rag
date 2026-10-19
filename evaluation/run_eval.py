"""
评估 CLI 入口。

用法：
    # 真实评估（离线 + LLM 评委）
    python -m evaluation.run_eval --mode offline --use-llm

    # 快速自检（零依赖，验证指标引擎）
    python -m evaluation.run_eval --mode selftest

    # 抽样评估（前 3 条）
    python -m evaluation.run_eval --mode offline --n 3

    # 与纯 LLM 基线对比
    python -m evaluation.run_eval --mode offline --compare
"""
import argparse
import json
import sys
from pathlib import Path

# 保证 evaluation 目录在 sys.path（支持 python -m evaluation.run_eval 与直接运行）
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runner import run  # noqa: E402
from report import save_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="跨境电商 RAG 系统评估")
    parser.add_argument("--mode", default="selftest",
                        choices=["offline", "online", "selftest"], help="运行模式")
    parser.add_argument("--use-llm", action="store_true", help="启用 LLM 评委")
    parser.add_argument("--heuristic-only", action="store_true",
                        help="仅启发式评委（等价于不传 --use-llm）")
    parser.add_argument("--n", type=int, default=0, help="抽样条数（0=全量）")
    parser.add_argument("--compare", action="store_true", help="与纯 LLM 基线对比")
    parser.add_argument("--base-url", default="http://localhost:8001", help="online 模式服务地址")
    parser.add_argument("--testset", default="", help="自定义测试集路径（jsonl）")
    parser.add_argument("--no-save", action="store_true", help="不落盘报告")
    args = parser.parse_args()

    use_llm = args.use_llm and not args.heuristic_only
    testset = Path(args.testset) if args.testset else None

    print(f"[eval] 模式={args.mode}, LLM评委={use_llm}, 抽样={args.n or '全量'}, 对比基线={args.compare}")
    report = run(
        mode=args.mode,
        use_llm=use_llm,
        base_url=args.base_url,
        testset_path=testset,
        n=args.n if args.n > 0 else None,
        compare=args.compare,
    )

    overall = report.get("overall", {})
    print(f"[eval] 总分={overall.get('overall_score')} 等级={overall.get('grade')} "
          f"（检索={overall.get('retrieval_score')} 生成={overall.get('generation_score')}）")

    if not args.no_save:
        md_path, json_path = save_report(report)
        print(f"[eval] 报告已保存：\n  {md_path}\n  {json_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
