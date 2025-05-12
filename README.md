<div align="center">

# CSM: Cloud Sentiment Model

基于BERT和云模型的不确定性情感识别模型（南京邮电大学不确定性人工智能大作业）<br>
输入一段文字，模型预测该段文字是积极还是消极情感，并给出不确定度

</div>

## 环境配置

1. 克隆仓库并安装依赖，推荐使用 python3.10

```bash
git clone url/to/repo
cd dir/to/repo
conda create -n csm python=3.10 -y
conda activate csm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

2. 从 Huggingface 下载预训练的BERT模型 [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/tree/main)，按照下面的文件夹结构放置到 `pretrain` 文件夹下。只需要下载以下文件即可，其余文件不必下载。

```bash
# 在pretrain文件夹下需要新建文件夹chinese-roberta-wwm-ext-large
pretrain
    └─chinese-roberta-wwm-ext-large
        │-─ added_tokens.json
        │-─ config.json
        │-─ pytorch_model.bin
        │-─ special_tokens_map.json
        │-─ tokenizer.json
        │-─ tokenizer_config.json
        └-─ vocab.txt
```

## 数据集制作&修改配置文件

1. 你可以使用多个 `.csv` 或 `.tsv` 文件作为数据集。将他们存放到同一个文件夹下，然后在配置文件 `configs/config.yaml` 中修改 `data.path` 为你的数据集存放路径即可。
2. 你的 `.csv` 或 `.tsv` 格式的数据集必须是以下面的格式（以csv为例），其中 `label` 列的值必须是 `0` 或 `1`，表示消极或积极情感。`text` 列的值是训练文本。

```csv
label,text
1,帅的呀，就是越来越爱你！[爱你][爱你][爱你]
1,美~~~~~[爱你]
1,梦想有多大，舞台就有多大![鼓掌]
0,写点儿字容易吗？降税降税[怒骂]
0,可惜啊！它们不是我的啊！[泪][泪]800000啊
```

3. 按需修改配置文件，配置文件位于 `configs/config.yaml`

## 训练

1. 使用下面的命令开始训练，第一次开始训练时，会进行数据预处理（速度较慢）。之后如果继续训练，会直接加载上一次预处理后的数据。

```bash
python train.py -c path/to/config.yaml
```

2. 如果需要继续训练，可以使用`-m`传入初始模型，例如

```bash
python train.py -c path/to/config.yaml -m path/to/last/model.ckpt
```

## 推理

推理使用 `inference.py`，该脚本有两种模式。一次性推理和对话类的交互推理。

1. 一次性推理，使用下面的命令，其中 `-m` 传入训练好的模型，`-t` 传入需要推理的文本。注意如果文本中带有空格则需要使用引号将文本括起来。

```bash
python inference.py -m path/to/last/model.ckpt -t 你好呀，今天天气真好！

# 输出结果
Loading tokenizer and model...
[Input]: 你好呀，今天天气真好！
[Prediction]: Positive
[Uncertainty]: 0.2147147823125124
```

2. 对话类的交互推理，使用下面的命令，其中 `-m` 传入训练好的模型。不需要输入文本。如果想退出聊天，按下 `Ctrl+C` 即可。

```bash
python inference.py -m path/to/last/model.ckpt

# 输出结果
Loading tokenizer and model...
>>> 你好呀，今天天气真好！
Prediction: Positive, Uncertainty: 0.2147147823125124
>>> 我太难受了！          
Prediction: Negative, Uncertainty: 0.08376982249319553
```

3. 若需要指定最大推理文本长度，可以传入参数 `--max_length`，默认为128。

## 参考

- 中文 BERT-wwm | [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
- 中文 自然语言处理 语料/数据集 | [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
