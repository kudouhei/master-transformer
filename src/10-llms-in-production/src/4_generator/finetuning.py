import os 
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from datasets import load_dataset

dataset = load_dataset("text", data_files="./data/crimeandpunishment.txt")
dataset = dataset.filter(lambda sentence: len(sentence["text"]) > 1)
print(dataset["train"][0])

# create model directory to save the model
model_dir = "./models/fineGPT/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# # Establish our GPT2 parameters (different from the paper and scratchGPT)
config = GPT2Config(
    vocab_size=50261,
    n_positions=256,
    n_embd=768,
    activation_function="gelu",
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = {"bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    "mask_token": "<MASK>",}

tokenizer.add_special_tokens(special_tokens)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config, ignore_mismatched_sizes=True)

# Create a tokenize function
def tokenize(batch):
    return tokenizer(str(batch), padding="max_length", truncation=True, max_length=256)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=False)

# Create a data collator
# Masked Language Modeling - adds <MASK> tokens to guess the words
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Create a training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
    report_to="none",
)

# Create a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

# Train the model
trainer.train()
trainer.save_model(model_dir)

tokenizer.save_pretrained(model_dir)

# load the saved model
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Test the saved model
input = "To be or not"
tokenized_inputs = tokenizer(input, return_tensors="pt")

out = model.generate(
    input_ids=tokenized_inputs["input_ids"],
    attention_mask=tokenized_inputs["attention_mask"],
    max_length=256,
    num_beams=5,
    temperature=0.7,
    top_k=50,
    top_p=0.90,
    no_repeat_ngram_size=2,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
