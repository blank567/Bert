"""从已有中文 BERT 权重微调标点分类头；全体 BERT 参数参与更新。"""

from bert_punct.runtime import configure_storage, configure_threads
configure_storage()

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from bert_punct.encoding import CharacterEncoder
from bert_punct.inference import BatchPunctuator
from bert_punct.schema import LABELS, LABEL2ID, ID2LABEL
from bert_punct.training import (read_records, WindowDataset, TrainingCollator, select_device,
                                 set_seed, evaluate_records, total_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/bert-base-chinese")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-overlap", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 表示按 epochs 完整训练")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--train-articles", type=int, default=0, help="仅用于短程验证，0 为全部")
    parser.add_argument("--val-articles", type=int, default=0, help="0 为全部；不得用测试集选模型")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.grad_accum, args.eval_every, args.threads) < 1:
        parser.error("轮数、batch、累积步数、验证间隔和线程数必须为正数")
    if min(args.train_articles, args.val_articles, args.max_steps) < 0:
        parser.error("样本/步数限制不能为负数")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("warmup-ratio 必须位于 [0,1]")
    if args.max_length < 3 or not 0 <= args.eval_overlap < args.max_length - 2:
        parser.error("max-length 至少为 3，且 0 <= eval-overlap < max-length - 2")
    if (args.output_dir / "run_config.json").exists():
        parser.error("输出目录已有实验记录，请使用新的 output-dir，以免混合日志和旧 checkpoint")
    configure_threads(args.threads)
    set_seed(args.seed)
    device = select_device(args.device)
    if args.fp16 and device.type != "cuda":
        parser.error("--fp16 仅适用于 CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(LABELS),
                                                           id2label=ID2LABEL, label2id=LABEL2ID)
    if args.max_length > model.config.max_position_embeddings:
        parser.error("max-length 超过模型位置编码上限（含特殊 Token）")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    encoder = CharacterEncoder(tokenizer)
    training_rows = read_records(args.data_dir / "train.jsonl", args.train_articles)
    validation_rows = read_records(args.data_dir / "val.jsonl", args.val_articles)
    dataset = WindowDataset(training_rows, encoder, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=TrainingCollator(tokenizer.pad_token_id))
    updates = total_steps(len(dataset), args.batch_size, args.grad_accum, args.epochs, args.max_steps)
    # bias 和 LayerNorm 权重不施加 weight decay。
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        (undecayed if name.endswith("bias") or "LayerNorm.weight" in name else decayed).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decayed, "weight_decay": args.weight_decay},
                                  {"params": undecayed, "weight_decay": 0.0}], lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * args.warmup_ratio), updates)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    punctuator = BatchPunctuator(model, tokenizer, device=device, fp16=args.fp16, id2label=ID2LABEL)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    step, best_f1, losses = 0, -1.0, []
    began = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(args.epochs):
        for index, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # 最后一个累积组可能不足 grad_accum，按实际组大小归一化。
            group_start = (index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                loss = model(**batch).loss
            losses.append(float(loss.detach()))
            scaler.scale(loss / group_size).backward()
            if (index + 1) % args.grad_accum and index + 1 != len(loader):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # 溢出导致本次参数未更新时，也不推进学习率或 optimizer step 计数。
            did_update = scaler.get_scale() >= previous_scale
            optimizer.zero_grad(set_to_none=True)
            if not did_update:
                continue
            scheduler.step()
            step += 1
            log = {"step": step, "epoch": epoch + 1, "loss": sum(losses) / len(losses),
                   "lr": scheduler.get_last_lr()[0], "elapsed_s": time.perf_counter() - began}
            losses.clear()
            print(json.dumps(log), flush=True)
            if step % args.eval_every == 0 or step == updates:
                report = evaluate_records(punctuator, validation_rows, batch_size=args.eval_batch_size,
                                          max_length=args.max_length, overlap=args.eval_overlap)
                log["validation"] = report
                f1 = report["punctuation_micro"]["f1"]
                if f1 > best_f1:
                    best_f1 = f1
                    model.save_pretrained(args.output_dir / "best")
                    tokenizer.save_pretrained(args.output_dir / "best")
                    (args.output_dir / "best_validation.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"step": step, "validation_micro_f1": f1, "best_f1": best_f1}), flush=True)
                model.train()
            with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(log, ensure_ascii=False) + "\n")
            if step >= updates:
                break
        if step >= updates:
            break
    if step < updates:
        raise RuntimeError(f"仅完成 {step}/{updates} 次有效更新；可能连续发生 AMP 溢出。请检查精度和学习率，不能视为训练完成")
    summary = {"optimizer_steps": step, "planned_steps": updates, "training_articles": len(training_rows),
               "training_windows": len(dataset), "validation_articles": len(validation_rows),
               "best_validation_micro_f1": best_f1, "elapsed_s": time.perf_counter() - began,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device),
               "purpose": "short_validation" if args.train_articles or args.max_steps else "full_training",
               "test_used_for_model_selection": False}
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
