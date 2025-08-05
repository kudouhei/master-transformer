## Unsloth
Unsloth is a new LLM based on the LLaMA architecture. It is a fine-tuned version of LLaMA with 1.1B parameters. It aims to be a general-purpose LLM that can be used for a wide range of tasks, improving the training efficiency and the quality of the model.
- **Kernel-level optimization**: Fused operations (such as LayerNorm + RMSNorm) to reduce GPU memory access
- **Automatic precision management**: Intelligent mixed precision (FP16/BF16) and gradient checkpointing
- **Efficient attention mechanisms**: Integration of Flash Attention 2/3 and Sliding Window Attention
- **Quantization support**: 4-bit/8-bit loading (compatible with AWQ/GPTQ)
- **Streamlined architecture**: Removal of redundant layers (such as unnecessary Embedding layers)

### Install
```bash
pip install unsloth
```

### Usage
```python
from unsloth import FastLanguageModel
import torch

# 加载模型（4-bit 量化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# 配置 LoRA 微调
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA 秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
)

# 准备数据
inputs = tokenizer(["<|im_start|>user\nHello!<|im_end|>"], return_tensors="pt")

# 训练（比普通训练快 2-5 倍）
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()
```

### Core code analysis
path: `unsloth/kernels/layer_norm.py`

**1. Kernel Fusion Optimization: `layer_norm.py`**






