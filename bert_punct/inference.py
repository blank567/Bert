"""独立重叠窗口、跨窗口 batch 前向与逐字符中心加权融合。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence, TypeVar

import torch

from .encoding import CharacterEncoder, EncodedWindow
from .schema import ID2LABEL


@dataclass
class PredictionResult:
    text: str
    labels: list[str]
    punctuated_text: str
    stats: dict[str, Any]


_Item = TypeVar("_Item")


def _adaptive_batches(
    items: Sequence[_Item],
    batch_size: int,
    forward: Callable[[Sequence[_Item]], torch.Tensor],
    *,
    allow_cuda_oom: bool,
    on_cuda_oom: Callable[[], None],
) -> Iterator[tuple[Sequence[_Item], torch.Tensor]]:
    """只有真实 CUDA OOM 才减半重试；成功后才推进游标，避免漏窗或重算成功窗。"""
    index = 0
    current_batch_size = batch_size
    while index < len(items):
        batch = items[index : index + current_batch_size]
        try:
            output = forward(batch)
        except torch.cuda.OutOfMemoryError:
            if not allow_cuda_oom or len(batch) == 1:
                raise
            current_batch_size = max(1, len(batch) // 2)
        else:
            yield batch, output
            index += len(batch)
            continue
        # 离开 except 后，失败前向的 traceback 可释放，随后清理 CUDA 缓存。
        on_cuda_oom()


class BatchPunctuator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str | torch.device | None = None,
        fp16: bool = False,
        id2label: dict[int, str] | None = None,
    ) -> None:
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.encoder = CharacterEncoder(tokenizer)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.id2label = {int(key): str(value) for key, value in (id2label or ID2LABEL).items()}
        if set(self.id2label) != set(range(len(self.id2label))) or "O" not in self.id2label.values():
            raise ValueError("id2label 必须从 0 连续编号并包含无标点标签 O")
        num_labels = getattr(model.config, "num_labels", len(self.id2label))
        if num_labels != len(self.id2label):
            raise ValueError("模型标签数量与标点标签映射不一致")
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("tokenizer 必须提供 pad_token_id，用于动态 padding")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward(self, jobs: Sequence[tuple[int, EncodedWindow]]) -> torch.Tensor:
        lengths = [len(window.input_ids) for _, window in jobs]
        # input_ids / attention_mask: [当前 batch 大小, 当前 batch 最长窗口]。
        input_ids = torch.full(
            (len(jobs), max(lengths)), int(self.tokenizer.pad_token_id), dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, (_, window) in enumerate(jobs):
            input_ids[row, : lengths[row]] = torch.tensor(window.input_ids, dtype=torch.long)
            attention_mask[row, : lengths[row]] = 1
        mixed_precision = torch.autocast("cuda", dtype=torch.float16) if self.fp16 else nullcontext()
        with torch.inference_mode(), mixed_precision:
            result = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
            logits = result.logits if hasattr(result, "logits") else result["logits"]
            if logits.ndim != 3 or logits.shape[:2] != input_ids.shape or logits.shape[2] != len(self.id2label):
                raise ValueError("模型 logits 应为 [batch, sequence, 标点标签数]")
            # 只保留当前 batch 的 CPU float32 logits；GPU 上不累积历史窗口。
            return logits.detach().float().cpu()

    def predict(
        self, text: str, *, batch_size: int = 8, max_length: int = 512, overlap: int = 100
    ) -> PredictionResult:
        return self.predict_many([text], batch_size=batch_size, max_length=max_length, overlap=overlap)[0]

    def predict_many(
        self, texts: Sequence[str], *, batch_size: int = 8, max_length: int = 512, overlap: int = 100
    ) -> list[PredictionResult]:
        if isinstance(texts, str):
            raise TypeError("predict_many 需要字符串序列；单个字符串请使用 predict")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        model_limit = getattr(self.model.config, "max_position_embeddings", 512)
        if max_length > model_limit:
            raise ValueError(f"max_length={max_length} 超过模型位置上限 {model_limit}")
        self.model.eval()
        self._sync()
        started = perf_counter()
        jobs: list[tuple[int, EncodedWindow]] = []
        window_counts = [0] * len(texts)
        token_counts = [0] * len(texts)
        for doc_id, text in enumerate(texts):
            for window in self.encoder.iter_windows(text, max_length=max_length, overlap=overlap):
                jobs.append((doc_id, window))
                window_counts[doc_id] += 1
                token_counts[doc_id] += len(window.input_ids)
        # CPU 累加器: [原文字符数, 标签数]，每个原文位置最终只输出一次。
        sums = [torch.zeros((len(text), len(self.id2label)), dtype=torch.float32) for text in texts]
        weights = [torch.zeros(len(text), dtype=torch.float32) for text in texts]
        forward_attempts = 0
        oom_retries = 0
        effective_batch_sizes: list[int] = []
        padded_tokens = 0

        def forward(batch: Sequence[tuple[int, EncodedWindow]]) -> torch.Tensor:
            nonlocal forward_attempts
            forward_attempts += 1
            return self._forward(batch)

        def recover_cuda_oom() -> None:
            nonlocal oom_retries
            oom_retries += 1
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        for batch, logits in _adaptive_batches(
            jobs, batch_size, forward,
            allow_cuda_oom=self.device.type == "cuda", on_cuda_oom=recover_cuda_oom,
        ):
            effective_batch_sizes.append(len(batch))
            padded_tokens += len(batch) * max(len(window.input_ids) for _, window in batch)
            for row, (doc_id, window) in enumerate(batch):
                positions = torch.tensor(window.char_positions, dtype=torch.long)
                # 离两端越远权重越大；首尾权重大于零，因此任何字符都有覆盖。
                center_weights = torch.minimum(positions, len(window.input_ids) - 1 - positions).float()
                selected = logits[row, positions]
                target = slice(window.char_start, window.char_end)
                sums[doc_id][target] += selected * center_weights[:, None]
                weights[doc_id][target] += center_weights
        self._sync()
        elapsed = perf_counter() - started
        results: list[PredictionResult] = []
        for doc_id, text in enumerate(texts):
            if len(text) and not bool(torch.all(weights[doc_id] > 0)):
                raise RuntimeError("滑窗存在未覆盖字符")
            label_ids = sums[doc_id].argmax(dim=-1).tolist()
            labels = [self.id2label[label_id] for label_id in label_ids]
            punctuated = "".join(char + ("" if label == "O" else label) for char, label in zip(text, labels))
            results.append(PredictionResult(
                text=text,
                labels=labels,
                punctuated_text=punctuated,
                stats={
                    "characters": len(text),
                    "windows": window_counts[doc_id],
                    "input_tokens": token_counts[doc_id],
                    "batch_size_requested": batch_size,
                    "effective_batch_sizes": effective_batch_sizes.copy(),
                    "forward_calls": len(effective_batch_sizes),
                    "forward_attempts": forward_attempts,
                    "oom_retries": oom_retries,
                    "padded_tokens": padded_tokens,
                    "elapsed_seconds": elapsed,
                    "characters_per_second": sum(map(len, texts)) / elapsed if elapsed else 0.0,
                    "document_count": len(texts),
                    "statistics_scope": "characters/windows/input_tokens are per document; timing/calls/batches are shared by the entire predict_many call",
                    "device": str(self.device),
                    "fp16": self.fp16,
                },
            ))
        # 将标签选择与文本还原也纳入整次调用的端到端统计。
        elapsed = perf_counter() - started
        for result in results:
            result.stats["elapsed_seconds"] = elapsed
            result.stats["characters_per_second"] = sum(map(len, texts)) / elapsed if elapsed else 0.0
        return results
