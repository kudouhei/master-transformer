# 训练两种不同的分词器
# BPE vs SentencePiece：
# - BPE：需要预分词，空格用 Ġ 表示。
# - SentencePiece：直接处理原始文本，空格用 ▁ 表示。
# 保存格式：
# - 原始分词器：保存 vocab.json 和 merges.txt。
# - Hugging Face 格式：保存为 tokenizer_config.json + 其他文件。

import os
from pathlib import Path

import transformers
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing

# 获取所有 .txt 文件的路径
# initialize the texts to train from
paths = [str(x) for x in Path("./data/").glob("**/*.txt")]

# Train a Byte-Pair Encoding tokenizer
bpe_tokenizer = ByteLevelBPETokenizer()

bpe_tokenizer.train(
    files=paths,
    vocab_size=52_000,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)

token_dir = "./models/tokenizers/bytelevelbpe/"

if not os.path.exists(token_dir):
    os.makedirs(token_dir)
bpe_tokenizer.save_model(token_dir)

# 加载训练好的 BPE 分词器
bpe_tokenizer = ByteLevelBPETokenizer(
    f"{token_dir}vocab.json",
    f"{token_dir}merges.txt",
)

example_text = "This sentence is getting encoded by a tokenizer."
print(bpe_tokenizer.encode(example_text).tokens)

print(bpe_tokenizer.encode(example_text).ids)

#['This', 'Ġsentence', 'Ġis', 'Ġgetting', 'Ġenc', 'od', 'ed', 'Ġby', 'Ġa', 'Ġtoken', 'iz', 'er', '.']
# [812, 3859, 344, 2424, 3337, 494, 287, 440, 263, 12007, 947, 275, 18]

bpe_tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", bpe_tokenizer.token_to_id("</s>")),
    ("<s>", bpe_tokenizer.token_to_id("<s>")),
)
bpe_tokenizer.enable_truncation(max_length=512)

# Train a Sentencepiece Tokenizer
special_tokens = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<cls>",
    "<sep>",
    "<mask>",
]
sentencepiece_tokenizer = SentencePieceBPETokenizer()

sentencepiece_tokenizer.train(
    files=paths,
    vocab_size=4000,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens,
)

token_dir = "./models/tokenizers/sentencepiece/"
if not os.path.exists(token_dir):
    os.makedirs(token_dir)
sentencepiece_tokenizer.save_model(token_dir)

# convert
tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_object=sentencepiece_tokenizer,
    model_max_length=512,
    special_tokens=special_tokens,
)
tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = sentencepiece_tokenizer.token_to_id("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = sentencepiece_tokenizer.token_to_id("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = sentencepiece_tokenizer.token_to_id("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = sentencepiece_tokenizer.token_to_id("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = sentencepiece_tokenizer.token_to_id("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = sentencepiece_tokenizer.token_to_id("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = sentencepiece_tokenizer.token_to_id("<mask>")
# and save for later!
tokenizer.save_pretrained(token_dir)

print(tokenizer.tokenize(example_text))

print(tokenizer.encode(example_text))

#['▁This', '▁sent', 'ence', '▁is', '▁getting', '▁enc', 'od', 'ed', '▁by', '▁a', '▁to', 'ken', 'iz', 'er.']
#[734, 1255, 478, 209, 2409, 3299, 370, 145, 318, 119, 143, 1548, 834, 3693]