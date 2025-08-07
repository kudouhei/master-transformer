import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# 加载数据集（GLUE SST-2 情感分析任务）
dataset = load_dataset("glue", "sst2")

# 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,                  # LoRA 的秩（rank）
    lora_alpha=16,        # 缩放因子
    target_modules=["query", "value"],  # 针对 BERT 的注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLASSIFICATION"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数（仅 LoRA 参数）

# 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./models/LoRA_peft/lora_finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 训练！
trainer.train()

# 保存 LoRA 适配器（仅保存少量参数）
model.save_pretrained("./models/LoRA_peft/lora_adapter")