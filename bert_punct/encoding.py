"""按字符对齐 BERT 子词，保证训练标签与滑窗推理采用同一套位置规则。"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class EncodedText:
    text: str
    token_ids: list[int]
    # 均为没有 [CLS]/[SEP] 时的位置；end 是最后一个子词的闭区间下标。
    char_token_starts: list[int]
    char_token_ends: list[int]


@dataclass
class EncodedWindow:
    char_start: int
    char_end: int
    input_ids: list[int]
    # 每个字符只监督最后一个子词；位置已经包含开头的 [CLS] 偏移。
    char_positions: list[int]


class CharacterEncoder:
    """字符可能对应多个子词，不能用 ``len(text)`` 代替实际 token 数。"""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.cls_token_id = getattr(tokenizer, "cls_token_id", None)
        self.sep_token_id = getattr(tokenizer, "sep_token_id", None)
        self.unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if self.cls_token_id is None or self.sep_token_id is None:
            raise ValueError("BERT tokenizer 必须提供 cls_token_id 和 sep_token_id")
        if self.unk_token_id is None:
            raise ValueError("BERT tokenizer 必须提供 unk_token_id，以保留未识别字符")
        self._cache: dict[str, tuple[int, ...]] = {}

    def _character_ids(self, char: str) -> tuple[int, ...]:
        if char not in self._cache:
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            # 空白、控制字符也保留一个位置，输出才能逐字符对齐原文。
            self._cache[char] = tuple(int(value) for value in ids) or (int(self.unk_token_id),)
        return self._cache[char]

    def encode(self, text: str) -> EncodedText:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        token_ids: list[int] = []
        starts: list[int] = []
        ends: list[int] = []
        for char in text:
            starts.append(len(token_ids))
            token_ids.extend(self._character_ids(char))
            ends.append(len(token_ids) - 1)
        return EncodedText(text, token_ids, starts, ends)

    def iter_windows(
        self, text: str, max_length: int = 512, overlap: int = 100
    ) -> Iterator[EncodedWindow]:
        """窗口长度含两个特殊 token；overlap 是最多重叠的内容 token 数。"""
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 3:
            raise ValueError("max_length 必须是至少为 3 的整数（包含 [CLS] 和 [SEP]）")
        budget = max_length - 2
        if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < budget:
            raise ValueError("overlap 必须是整数，且满足 0 <= overlap < max_length - 2")
        encoded = self.encode(text)
        # prefix[i] 表示前 i 个字符占用多少 token，窗口绝不切开一个字符。
        prefix = [0] + [end + 1 for end in encoded.char_token_ends]
        for index in range(len(text)):
            if prefix[index + 1] - prefix[index] > budget:
                raise ValueError(f"第 {index} 个字符的子词数超过窗口内容预算 {budget}")
        start = 0
        while start < len(text):
            end = bisect_right(prefix, prefix[start] + budget, lo=start + 1) - 1
            content = encoded.token_ids[prefix[start] : prefix[end]]
            positions = [value - prefix[start] + 1 for value in encoded.char_token_ends[start:end]]
            yield EncodedWindow(
                char_start=start,
                char_end=end,
                input_ids=[int(self.cls_token_id), *content, int(self.sep_token_id)],
                char_positions=positions,
            )
            if end == len(text):
                break
            # lo=start+1 强制向前移动；遇到长子词字符时允许减少重叠。
            start = bisect_left(prefix, prefix[end] - overlap, lo=start + 1, hi=end + 1)

    def make_windows(
        self, text: str, max_length: int = 512, overlap: int = 100
    ) -> list[EncodedWindow]:
        return list(self.iter_windows(text, max_length=max_length, overlap=overlap))
