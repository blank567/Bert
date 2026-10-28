"""同权重、同文本的串行/批量滑窗基准；CUDA 计时同步，含预热与重复。"""

from bert_punct.runtime import configure_storage, configure_threads
configure_storage()

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from predict import load_punctuator
from bert_punct.baseline import sequential_predict
from bert_punct.metrics import PunctuationMetrics
from bert_punct.training import read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    parser.add_argument("--characters", type=int, default=4000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--max-length", type=int, default=402, help="含 CLS/SEP；402 对应400个内容 token")
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--commit-chars", type=int, default=300)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-sequential", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.characters < 1 or args.repeats < 1 or args.warmups < 0 or min(args.batch_sizes) < 1:
        parser.error("字符数、重复次数、batch 必须为正数，预热次数不能为负数")
    configure_threads(args.threads)
    rows = read_records(args.data)
    row = max(rows, key=lambda r: len(r["text"]))
    text, gold = row["text"][:args.characters], row["labels"][:args.characters]
    punctuator = load_punctuator(args.model, args.device, args.fp16)
    # 共享 tokenizer 字符缓存；所有算法都先完成字符编码后再预热。
    punctuator.encoder.encode(text)
    runs = []
    predictions = {}
    strategies = [("fixed_windows", batch) for batch in sorted(set([1] + args.batch_sizes))]
    if args.include_sequential:
        strategies.append(("predicted_sentence_serial", 1))
    for strategy, batch_size in strategies:
        def run_once():
            if strategy == "predicted_sentence_serial":
                return sequential_predict(punctuator, text, max_length=args.max_length, commit_chars=args.commit_chars)
            return punctuator.predict(text, batch_size=batch_size, max_length=args.max_length, overlap=args.overlap)
        for _ in range(args.warmups):
            run_once()
        if punctuator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(punctuator.device)
        durations = []
        for _ in range(args.repeats):
            punctuator._sync()
            started = time.perf_counter()
            result = run_once()
            punctuator._sync()
            durations.append(time.perf_counter() - started)
        metrics = PunctuationMetrics()
        metrics.update(gold, result.labels)
        checksum = hashlib.sha256("".join(result.labels).encode()).hexdigest()
        if strategy == "fixed_windows":
            predictions[batch_size] = result.labels
        report = {"strategy": strategy, "batch_size": batch_size,
                  "median_seconds": statistics.median(durations), "seconds": durations,
                  "characters_per_second": len(text) / statistics.median(durations),
                  "windows": result.stats["windows"], "forward_calls": result.stats["forward_calls"],
                  "input_tokens": result.stats["input_tokens"], "checksum": checksum,
                  "punctuation_micro": metrics.compute()["punctuation_micro"],
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(punctuator.device)
                  if punctuator.device.type == "cuda" else None,
                  "effective_batch_sizes": result.stats.get("effective_batch_sizes"),
                  "padded_tokens": result.stats.get("padded_tokens"),
                  "forward_attempts": result.stats.get("forward_attempts", result.stats["forward_calls"]),
                  "oom_retries": result.stats.get("oom_retries", 0)}
        runs.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    single = next(r for r in runs if r["strategy"] == "fixed_windows" and r["batch_size"] == 1)
    for run in runs:
        run["speedup_vs_fixed_batch1"] = single["median_seconds"] / run["median_seconds"]
        if run["strategy"] == "fixed_windows":
            different = sum(a != b for a, b in zip(predictions[1], predictions[run["batch_size"]]))
            run["label_differences_vs_batch1"] = different
            run["exact_label_parity"] = different == 0
    report = {"model": args.model, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": str(punctuator.device), "gpu": torch.cuda.get_device_name(punctuator.device)
              if punctuator.device.type == "cuda" else None, "cpu": platform.processor(),
              "threads": args.threads, "fp16": punctuator.fp16, "article_id": row["id"],
              "characters": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "max_length": args.max_length, "overlap_tokens": args.overlap,
              "sequential_commit_chars": args.commit_chars,
              "warmups": args.warmups, "repeats": args.repeats,
              "timing_scope": "warm-cache end-to-end including encoding, transfer, forward, fusion and rendering; excludes model loading",
              "limitation": "CPU results do not establish GPU speedup; sentence-based baseline has different contexts",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
