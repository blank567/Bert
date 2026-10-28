"""在导入 Hugging Face / PyTorch 之前将缓存和临时文件固定到 E 盘。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_storage() -> None:
    # 使用项目目录，避免依赖用户原先指向 C 盘的缓存设置。
    paths = {
        "HF_HOME": ROOT / "cache" / "huggingface",
        "HF_HUB_CACHE": ROOT / "cache" / "huggingface" / "hub",
        "TORCH_HOME": ROOT / "cache" / "torch",
        "XDG_CACHE_HOME": ROOT / "cache",
        "TEMP": Path(r"E:\codex\tmp"),
        "TMP": Path(r"E:\codex\tmp"),
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_threads(count: int = 4) -> None:
    import torch

    # 小 batch 在 CPU 上常受过量线程调度影响，允许命令行调整。
    torch.set_num_threads(count)
