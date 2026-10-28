from types import SimpleNamespace

import pytest
import torch

from bert_punct.encoding import CharacterEncoder
from bert_punct.inference import BatchPunctuator, _adaptive_batches
from bert_punct.schema import ID2LABEL


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    unk_token_id = 0
    pad_token_id = 103

    def __init__(self):
        self.calls = []

    def encode(self, char, add_special_tokens=False):
        assert add_special_tokens is False
        self.calls.append(char)
        if char == "𠀀":
            return [21, 22, 23]
        if char == "界":
            return [30, 31]
        if char.isspace():
            return []
        return [8 + ord(char) % 16]


class FakeModel(torch.nn.Module):
    def __init__(self, mode="token", error=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(num_labels=len(ID2LABEL), max_position_embeddings=512)
        self.mode = mode
        self.error = error
        self.batches = []

    def forward(self, input_ids, attention_mask):
        if self.error is not None:
            raise self.error
        self.batches.append((input_ids.clone(), attention_mask.clone()))
        if self.mode == "position":
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            # 4 个内容 token 的窗口中，局部两端预测 O，中间预测逗号。
            label_ids = ((positions == 2) | (positions == 3)).long()[None].expand_as(input_ids)
        elif self.mode == "none":
            label_ids = torch.zeros_like(input_ids)
        else:
            label_ids = input_ids % len(ID2LABEL)
        logits = torch.nn.functional.one_hot(label_ids, num_classes=len(ID2LABEL)).float() * 9
        return SimpleNamespace(logits=logits + self.anchor)


def make_predictor(**kwargs):
    return BatchPunctuator(FakeModel(**kwargs), FakeTokenizer())


def test_alignment_uses_last_subtoken_and_preserves_empty_tokenization():
    tokenizer = FakeTokenizer()
    encoder = CharacterEncoder(tokenizer)
    encoded = encoder.encode("甲𠀀 界甲")
    assert encoded.char_token_starts == [0, 1, 4, 5, 7]
    assert encoded.char_token_ends == [0, 3, 4, 6, 7]
    window = encoder.make_windows("甲𠀀 界甲", max_length=12, overlap=0)[0]
    assert window.char_positions == [1, 4, 5, 7, 8]
    assert window.input_ids[0] == tokenizer.cls_token_id
    assert window.input_ids[-1] == tokenizer.sep_token_id
    assert window.input_ids[window.char_positions[2]] == tokenizer.unk_token_id
    assert tokenizer.calls.count("甲") == 1


@pytest.mark.parametrize("text", ["", "甲", "甲乙丙", "甲𠀀界乙" * 121, "\n甲 空\t白"])
@pytest.mark.parametrize("overlap", [0, 1, 4])
def test_window_budget_character_boundaries_progress_and_full_coverage(text, overlap):
    encoder = CharacterEncoder(FakeTokenizer())
    windows = encoder.make_windows(text, max_length=7, overlap=overlap)
    coverage = [0] * len(text)
    previous_start = -1
    for window in windows:
        assert len(window.input_ids) <= 7
        assert window.char_start > previous_start
        previous_start = window.char_start
        assert window.char_end > window.char_start
        assert len(window.char_positions) == window.char_end - window.char_start
        for char_id, position in zip(range(window.char_start, window.char_end), window.char_positions):
            coverage[char_id] += 1
            expected = encoder.encode(text[char_id]).token_ids[-1]
            assert window.input_ids[position] == expected
    assert all(count >= 1 for count in coverage)
    assert bool(windows) == bool(text)
    if windows:
        assert windows[0].char_start == 0
        assert windows[-1].char_end == len(text)


@pytest.mark.parametrize("max_length, overlap", [(2, 0), (10, -1), (10, 8), (5.5, 0), (6, True)])
def test_invalid_window_configuration(max_length, overlap):
    with pytest.raises(ValueError):
        CharacterEncoder(FakeTokenizer()).make_windows("甲乙", max_length=max_length, overlap=overlap)


def test_single_character_that_exceeds_token_budget_is_reported():
    with pytest.raises(ValueError, match="第 1 个字符"):
        CharacterEncoder(FakeTokenizer()).make_windows("甲𠀀", max_length=4, overlap=0)


@pytest.mark.parametrize("text", ["", "甲", "甲𠀀界乙" * 133 + "完", "\n空 白\t"])
def test_batch_one_and_batch_seven_have_identical_global_results(text):
    predictor = make_predictor()
    serial = predictor.predict(text, batch_size=1, max_length=19, overlap=5)
    batched = predictor.predict(text, batch_size=7, max_length=19, overlap=5)
    assert serial.labels == batched.labels
    assert serial.punctuated_text == batched.punctuated_text
    assert len(batched.labels) == len(text)
    expected_labels = [ID2LABEL[predictor.encoder.encode(char).token_ids[-1] % len(ID2LABEL)] for char in text]
    assert batched.labels == expected_labels
    assert batched.stats["forward_calls"] == (batched.stats["windows"] + 6) // 7
    assert batched.stats["forward_calls"] <= serial.stats["forward_calls"]


def test_triangular_fusion_prefers_center_context_across_seam():
    predictor = make_predictor(mode="position")
    result = predictor.predict("abcdef", batch_size=4, max_length=6, overlap=2)
    assert result.labels == ["O", "，", "，", "，", "，", "O"]
    assert result.punctuated_text == "ab，c，d，e，f"
    assert result.stats["windows"] == 2
    assert result.stats["forward_calls"] == 1


def test_predict_many_batches_across_documents_and_pads_only_to_current_maximum():
    predictor = make_predictor()
    results = predictor.predict_many(["甲乙", "", "𠀀", "终"], batch_size=8, max_length=32, overlap=4)
    assert [len(result.labels) for result in results] == [2, 0, 1, 1]
    assert len(predictor.model.batches) == 1
    inputs, mask = predictor.model.batches[0]
    assert tuple(inputs.shape) == (3, 5)
    assert mask.sum(dim=-1).tolist() == [4, 5, 3]
    assert torch.all(inputs[mask == 0] == predictor.tokenizer.pad_token_id)
    assert results[0].stats["padded_tokens"] == 15
    assert results[1].punctuated_text == ""


def test_empty_input_skips_model_and_no_final_period_is_forced():
    predictor = make_predictor(mode="none")
    assert predictor.predict("", max_length=16, overlap=0).stats["forward_calls"] == 0
    assert predictor.model.batches == []
    result = predictor.predict("天地玄黄", max_length=16, overlap=0)
    assert result.punctuated_text == "天地玄黄"
    assert not predictor.model.training


def test_context_limit_and_bad_batch_size_are_rejected():
    predictor = make_predictor()
    with pytest.raises(ValueError, match="位置上限"):
        predictor.predict("甲", max_length=513)
    with pytest.raises(ValueError, match="batch_size"):
        predictor.predict("甲", batch_size=0)
    with pytest.raises(TypeError, match="字符串序列"):
        predictor.predict_many("甲")


def test_cpu_fp16_request_stays_float32():
    predictor = BatchPunctuator(FakeModel(), FakeTokenizer(), device="cpu", fp16=True)
    result = predictor.predict("甲", max_length=8, overlap=0)
    assert result.stats["fp16"] is False


def test_adaptive_scheduler_retries_failed_batch_without_dropping_or_duplicating_windows():
    attempts = []
    recoveries = []

    def forward(batch):
        attempts.append(list(batch))
        if len(batch) > 2:
            # 只测试调度器，不假造 CUDA 设备，也不分配 GPU 内存。
            raise torch.cuda.OutOfMemoryError("simulated CUDA allocation failure")
        return torch.tensor(batch)

    outputs = list(_adaptive_batches(
        list(range(7)), 6, forward, allow_cuda_oom=True,
        on_cuda_oom=lambda: recoveries.append(True),
    ))
    assert attempts == [list(range(6)), [0, 1, 2], [0], [1], [2], [3], [4], [5], [6]]
    assert len(recoveries) == 2
    assert [item for batch, _ in outputs for item in batch] == list(range(7))
    assert [item for _, output in outputs for item in output.tolist()] == list(range(7))


def test_single_window_cuda_oom_and_unrelated_errors_propagate():
    def out_of_memory(batch):
        raise torch.cuda.OutOfMemoryError("one window still too large")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        list(_adaptive_batches([1], 1, out_of_memory, allow_cuda_oom=True, on_cuda_oom=lambda: None))
    predictor = make_predictor(error=RuntimeError("unrelated out of memory message"))
    with pytest.raises(RuntimeError, match="unrelated"):
        predictor.predict("甲乙", max_length=8, overlap=0)
    cpu_predictor = make_predictor(error=torch.cuda.OutOfMemoryError("CPU backend must not retry"))
    with pytest.raises(torch.cuda.OutOfMemoryError, match="CPU backend"):
        cpu_predictor.predict("甲乙", max_length=8, overlap=0)
