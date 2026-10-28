# BERT 中文断句与标点恢复

使用已有中文 BERT 权重微调逐字符标点分类器，并通过**固定重叠窗口、Batch 前向和全局位置融合**处理数千字长文。本项目对应现成语料微调与长文批量推理两项任务，无需重新预训练 BERT。

[实验报告（可单独提交）](EXPERIMENT_REPORT.md)包含原理、数据制备、参数、指标、结果分析和作业问题回答。

## 项目结果

以下为 2026-10-22 实施、2026-10-28 根据原始记录核对整理的结果。当前只完成短程微调，没有完成全量训练、完整二月评估或真实 GPU 实验。

| 内容 | 实测结果 |
|---|---|
| 数据制备 | 训练 2,817 篇、验证 313 篇、独立测试 2,623 篇；移除 5 篇精确重复文章 |
| 微调 | 加载真实中文 BERT 权重，使用 32 篇训练文章，完成 60 次参数更新 |
| 验证子集 | 8 篇、2,882 字符，最佳标点 micro F1 **51.28%** |
| 二月测试子集 | 8 篇、5,798 字符，标点 micro F1 **38.61%**、句界 F1 **3.51%** |
| 长文正确性 | 4,000 字完整保留；固定窗口 Batch=1/4/8 的输出标签完全一致 |
| 工程测试 | 54 项测试通过，覆盖语料、对齐、滑窗、融合、Batch 和串行基线 |

4,000 字使用 400 个内容 Token 的窗口、100 Token 重叠，共 13 个窗口。CPU 固定 4 线程，每种配置预热一次、正式测量三次：

| 推理方案 | 模型调用次数 | 耗时中位数 | 字符/秒 |
|---|---:|---:|---:|
| 固定窗口 Batch=1 | 13 | 6.033 秒 | 663.0 |
| 固定窗口 Batch=4 | 4 | 6.432 秒 | 621.8 |
| 固定窗口 Batch=8 | 2 | 6.418 秒 | 623.3 |
| 按预测句号推进的串行基线 | 13 | 6.499 秒 | 615.4 |

**Batch 减少模型调用次数，但本次 CPU 上没有观测到加速。** GPU 吞吐和显存尚未实测，不能由调用次数推导加速倍数。当前句界召回很低，模型属于流程验证模型；现代文指标不能证明古文效果。

## 方法

### 标点微调

移除目标标点，为每个字符预测其后的标点：

```text
原文：你好，世界。
输入：你 好 世 界
标签：O  ， O  。
```

标签为 `O / ， / 。 / ？ / ！ / ； / ： / 、`，`O` 表示不插入标点。模型结构为 `BERT → Dropout → Linear(768, 8)`，微调时更新全部参数；输入为 `[B,L]`，输出 logits 为 `[B,L,8]`。

字符可能对应多个子 Token，训练和推理均读取最后一个子 Token；其他子 Token、特殊 Token 和 padding 的训练标签为 `-100`。逐字符编码与整句 WordPiece 对英文单词的处理可能不同，必须使用同一编码器。当前只恢复七类标点，移除引号、括号等，连续目标标点按固定优先级保留一个。

### 长文 Batch 推理

```mermaid
flowchart LR
    A[无标点长文] --> B[按 Token 数切重叠窗口]
    B --> C[多个窗口组成 Batch]
    C --> D[BERT 前向]
    D --> E[按全局位置中心加权融合]
    E --> F[每个字符输出一个标点决策]
```

窗口位置提前确定，与预测句号无关，使单篇长文的多个窗口也可同时组批。`predict_many()` 支持跨文档窗口组批，采用动态 padding；尚未实现长度分桶和在线服务队列。

- `max_length` 包含 `[CLS]`、`[SEP]`，中文 BERT 上限 512 Token，正文最多 510 Token。
- `overlap` 以内容 Token 为单位；窗口边界不截断一个字符的子 Token。
- 重叠区域按全局字符位置做中心加权 logits 融合，统一输出以避免丢字、重字。
- 使用 `eval()`、`inference_mode()`；CUDA 可选 FP16，真实 CUDA OOM 时减半 Batch 并重试。
- 滑窗完整处理长文，但没有扩展单个窗口的上下文。

## 仓库结构与上传范围

```text
bert_punct/             核心实现：清洗、编码、微调辅助、推理、基线与指标
tests/                  工程测试
prepare_data.py         语料制备
download_model.py       获取预训练权重
train.py                微调与验证集选模型
evaluate.py             独立评估
predict.py              文本/文件推理
benchmark.py            Batch 与串行基准
requirements.txt        依赖范围
README.md               项目介绍与复现步骤
EXPERIMENT_REPORT.md    可提交的实验报告
```

`.gitignore` 排除 `docs/`、语料、模型权重、缓存、日志、预测文本、临时产物、虚拟环境和本地配置。仓库保留源代码、测试、依赖与这两份 Markdown。克隆后需自行准备语料、下载预训练权重并生成微调 checkpoint。

## 环境

实验使用 Windows、Python 3.10.19、PyTorch 2.10.0+cpu、Transformers 5.1.0 和 Conda `llm`。硬件为 RTX 3050 Laptop 4GB，但实验环境没有可用 CUDA，下面使用 CPU 复现。

在项目根目录的 PowerShell 执行：

```powershell
conda activate llm
$env:TEMP = 'E:\codex\tmp'
$env:TMP = 'E:\codex\tmp'
$env:PYTHONIOENCODING = 'utf-8'
python -m pip install --cache-dir .\cache\pip -r requirements.txt
```

若 Conda 未初始化，可用 `& E:\conda_envs\llm\python.exe` 代替 `python`。项目缓存写入当前 E 盘目录，临时目录为 `E:\codex\tmp`。CUDA 运行需要与驱动兼容的 CUDA 版 PyTorch；系统 Toolkit、驱动和 wheel 自带运行时是不同组件，安装 wheel 不需要同时修改系统 Toolkit。

## 快速复现

### 1. 数据与预训练模型

自行准备两份人民日报带音语料，放到 `data/raw/` 或修改为实际位置：

```powershell
python prepare_data.py `
  --train-source data/raw/1998-01-2003版-带音.txt `
  --test-source data/raw/1998-02-2003版-带音.txt `
  --output-dir data/processed
python download_model.py
```

严格读取 UTF-8/GB18030，去行号、分词空格和 `{pinyin}`，按文章合并。一月在正文精确去重后划分训练/验证，二月作独立测试。清洗不输入段落边界、不强制添加句号；含 `/n` 词性后缀的其他版本需扩展解析器。

### 2. 微调

```powershell
python train.py `
  --model models/bert-base-chinese --output-dir outputs/smoke `
  --device cpu --epochs 3 --batch-size 2 --grad-accum 1 `
  --max-length 128 --eval-overlap 32 `
  --train-articles 32 --val-articles 8 --max-steps 60 --eval-every 30
```

每 30 步验证，按排除 `O` 的标点 micro F1 保存 `outputs/smoke/best/`。测试集不参与选模型。`--epochs 3` 是轮数上限，本次在第一轮中达到 60 步后结束，并未完成三轮训练。

完整训练时将 `--train-articles`、`--val-articles`、`--max-steps` 保持默认 0，并调整长度、Batch 和轮数。CUDA 可用后可选 `--device cuda --fp16 --gradient-checkpointing`；训练和推理 Batch 需分别实测。当前不支持断点续训，每次实验应使用新的输出目录。

### 3. 评估与推理

```powershell
python evaluate.py `
  --model outputs/smoke/best --device cpu --max-articles 8 `
  --max-length 128 --overlap 32 --batch-size 4 `
  --output outputs/smoke/test_subset.json

python predict.py `
  --model outputs/smoke/best --device cpu `
  --text '今天天气很好我们出去走走吧' `
  --batch-size 4 --max-length 512 --overlap 100
```

正式测试应固定验证集选定的 checkpoint，取消 `--max-articles 8` 后评估完整二月集合。UTF-8 文件推理可用 `--input-file data/input.txt --output-file outputs/punctuated.txt`；已有标点、空格或注音可显式用 `--remove-existing-punctuation` 清洗。

### 4. 性能基准与测试

```powershell
python benchmark.py `
  --model outputs/smoke/best --device cpu --characters 4000 `
  --batch-sizes 1 4 8 --max-length 402 --overlap 100 `
  --warmups 1 --repeats 3 --include-sequential `
  --output outputs/smoke/benchmark_cpu.json

python -m pytest -q --basetemp E:\codex\course\Bert\artifacts\pytest-run
```

基准使用验证集中最长文章的前 4,000 字，固定模型和窗口配置，记录标签差异、实际批次、调用数和耗时。CUDA 环境另记录同步计时和峰值显存。运行负载和依赖版本变化可能导致结果不同。

## 参考

- [中文 BERT 预训练模型](https://huggingface.co/google-bert/bert-base-chinese)
- [Hugging Face Token Classification](https://huggingface.co/docs/transformers/main/tasks/token_classification)
- [PyTorch CUDA 说明](https://docs.pytorch.org/docs/stable/notes/cuda.html)
