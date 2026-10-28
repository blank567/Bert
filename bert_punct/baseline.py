"""作业中的串行基线：窗口完成后，依据已预测的句号决定下一个起点。"""

from bisect import bisect_right
from time import perf_counter
from .inference import PredictionResult


def sequential_predict(punctuator, text: str, *, max_length: int = 402, commit_chars: int = 300) -> PredictionResult:
    if commit_chars < 1 or max_length < 3:
        raise ValueError("commit_chars 必须为正数，max_length 至少为 3")
    # 一次编码以定位内容 token 预算，避免每次从头编码整个剩余长文。
    encoded = punctuator.encoder.encode(text)
    prefix = [0] + [position + 1 for position in encoded.char_token_ends]
    labels, starts = [], []
    calls = input_tokens = 0
    start = 0
    punctuator._sync()
    began = perf_counter()
    while start < len(text):
        end = bisect_right(prefix, prefix[start] + max_length - 2, lo=start + 1) - 1
        if end <= start:
            raise ValueError("单个字符的子词数已超过窗口内容预算")
        result = punctuator.predict(text[start:end], batch_size=1, max_length=max_length, overlap=0)
        calls += result.stats["forward_calls"]
        input_tokens += result.stats["input_tokens"]
        starts.append(start)
        if end == len(text):
            take = end - start
        else:
            limit = min(commit_chars, end - start)
            # 若前 300 字没有句号就推进 300 字，避免死循环；不人为插入句号。
            stops = [i + 1 for i, label in enumerate(result.labels[:limit]) if label == "。"]
            take = stops[-1] if stops else limit
        labels.extend(result.labels[:take])
        start += take
    punctuator._sync()
    elapsed = perf_counter() - began
    return PredictionResult(text, labels,
                            "".join(c + (label if label != "O" else "") for c, label in zip(text, labels)),
                            {"characters": len(text), "windows": calls, "forward_calls": calls,
                             "input_tokens": input_tokens, "starts": starts, "elapsed_seconds": elapsed,
                             "characters_per_second": len(text) / elapsed if elapsed else 0.0})
