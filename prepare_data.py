"""清洗人民日报：一月按文章分训练/验证，二月作为独立测试。"""

import argparse
import json
from pathlib import Path
from bert_punct.corpus import prepare_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = prepare_corpus(args.train_source, args.test_source, args.output_dir, args.validation_ratio, args.seed)
    # 文章 ID / 哈希的完整清单写入 stats.json，终端只显示摘要。
    summary = {"splits": {name: {k: v for k, v in values.items() if k not in ("ids", "content_hashes")}
                           for name, values in stats["splits"].items()},
               "audit_file": str(args.output_dir / "stats.json")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
