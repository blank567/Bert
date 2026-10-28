"""只评估指定 checkpoint，输出标点和句界 F1；本脚本不选模型、不更新权重。"""

from bert_punct.runtime import configure_storage, configure_threads
configure_storage()

import argparse
import json
from pathlib import Path
from predict import load_punctuator
from bert_punct.training import read_records, evaluate_records
from bert_punct.metrics import PunctuationMetrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--max-articles", type=int, default=0, help="0 为完整集合；非零只是流程验证")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_articles < 0:
        parser.error("max-articles 不能为负数")
    configure_threads(args.threads)
    records = read_records(args.data, args.max_articles)
    punctuator = load_punctuator(args.model, args.device, args.fp16)
    report = evaluate_records(punctuator, records, batch_size=args.batch_size,
                              max_length=args.max_length, overlap=args.overlap)
    baseline = PunctuationMetrics()
    for row in records:
        baseline.update(row["labels"], [0] * len(row["text"]))
    result = {"model": args.model, "data": str(args.data), "partial_evaluation": bool(args.max_articles),
              "article_ids": [row["id"] for row in records], "metrics": report,
              "always_O_baseline": baseline.compute(), "device": str(punctuator.device),
              "config": {"batch_size": args.batch_size, "max_length": args.max_length, "overlap": args.overlap}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": {k: v for k, v in report.items() if k not in ("confusion_matrix", "per_class")},
                      "partial_evaluation": result["partial_evaluation"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
