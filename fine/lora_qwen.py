import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model

# =============================
# 路径配置
# =============================
model_path = ""
data_path = "recipes_sft.jsonl"

# =============================
# tokenizer
# =============================
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

special_tokens = ["<|user|>", "<|assistant|>"]
tokenizer.add_special_tokens(
    {"additional_special_tokens": special_tokens}
)

# =============================
# dataset
# =============================
dataset = load_dataset("json", data_files=data_path)["train"]

# =============================
# tokenize
# =============================
def tokenize_function(example):

    messages = example["messages"]

    input_ids = []
    labels = []

    for msg in messages:

        if msg["role"] == "user":

            text = "<|user|>\n" + msg["content"] + "\n"

            tokenized = tokenizer(
                text,
                add_special_tokens=False
            )

            ids = tokenized["input_ids"]

            input_ids.extend(ids)
            labels.extend([-100] * len(ids))

        elif msg["role"] == "assistant":

            text = "<|assistant|>\n" + msg["content"] + "\n"

            tokenized = tokenizer(
                text,
                add_special_tokens=False
            )

            ids = tokenized["input_ids"]

            input_ids.extend(ids)
            labels.extend(ids)

    max_len = 2048

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    attention_mask = [1] * len(input_ids)

    padding_len = max_len - len(input_ids)

    if padding_len > 0:
        input_ids += [tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len
        attention_mask += [0] * padding_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names
)

# =============================
# model
# =============================
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=None,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model.resize_token_embeddings(len(tokenizer))


model.gradient_checkpointing_enable()

# =============================
# LoRA config
# =============================
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


model.enable_input_require_grads()

model.print_trainable_parameters()

# =============================
# training args
# =============================
training_args = TrainingArguments(
    output_dir="./qwen3-8b-lora",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,

    num_train_epochs=3,
    learning_rate=2e-4,

    fp16=True,

    logging_steps=10,
    save_steps=500,

    report_to="none",

    remove_unused_columns=False,

    dataloader_num_workers=0,

    ddp_find_unused_parameters=False
)

# =============================
# data collator（避免重复 padding）
# =============================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=False
)

# =============================
# Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()
