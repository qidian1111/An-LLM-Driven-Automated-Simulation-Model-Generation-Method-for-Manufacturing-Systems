# train_pos2conn.py
# 功能：微调 DeepSeek-R1-Distill-Qwen-14B，让它学会 POS -> CONN

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig,
)
from peft import LoraConfig, TaskType, get_peft_model
import torch

# ========== 1. 路径 & 基本配置 ==========

base_model_path = r""

DATA_PATH = r""
OUTPUT_DIR = r""

# ========== 2. 加载基座模型 ==========

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.half,
    device_map="auto",
)
# 训练阶段配合 gradient_checkpointing，显式关掉 use_cache（等价于 warning 里自动关）
model.config.use_cache = False

print("=== debug devices ===")
first_param = next(model.parameters())
print("First param device:", first_param.device)
print("=====================")

model.generation_config = GenerationConfig.from_pretrained(base_model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.enable_input_require_grads()

# ========== 3. 加载数据集 & 划分 train / eval ==========

print("===> loading ...")
df = pd.read_json(DATA_PATH)
full_ds = Dataset.from_pandas(df)
print("data num:", len(full_ds))

split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]
print("Tdata num:", len(train_ds))
print("Vdata num:", len(eval_ds))

# ========== 4. tokenizer & SYSTEM_PROMPT（专注连接关系） ==========

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    use_fast=False,
    trust_remote_code=True,
)
tokenizer.padding_side = "right"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

SYSTEM_PROMPT = (
    "You are an expert in manufacturing system layout analysis. Your task is to infer material-flow connection relationships among simulation entities based on structured equipment layout information. The inference process should consider equipment categories, spatial locations, port orientations, and manufacturing logistics constraints. Generate structured connection relationships according to the following rules: 1. The distance between the output port of an upstream equipment entity and the input port of a candidate downstream entity must be within the predefined spatial threshold; 2. The connection direction must follow the material flow principle, where the output port of an upstream entity can only connect to the input port of a downstream entity; 3. Valid connections require spatial consistency between corresponding port regions. If the connection regions of two entities do not overlap or intersect, no connection should be established; 4. When no candidate connection satisfies all predefined constraints, return an empty connection set rather than generating an artificial connection; 5. A single equipment entity may have multiple input or output connections; however, connections that violate equipment orientation or material flow logic must not be generated."
)


def process_func(example):
    MAX_LENGTH = 2048

    user_text = (example.get("instruction", "") or "") + "\n\n" + (
        example.get("input", "") or ""
    )
    assistant_text = example.get("output", "") or ""

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    instruction = tokenizer(
        prompt,
        add_special_tokens=False,
    )
    response = tokenizer(
        assistant_text,
        add_special_tokens=False,
    )

    input_ids = instruction["input_ids"] + response["input_ids"] + [
        tokenizer.eos_token_id
    ]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [
        tokenizer.eos_token_id
    ]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("===> token ...")
tokenized_train_ds = train_ds.map(process_func, remove_columns=train_ds.column_names)
tokenized_eval_ds = eval_ds.map(process_func, remove_columns=eval_ds.column_names)
print("Token finish：", len(tokenized_train_ds[0]["input_ids"]))

# ========== 5. 配置 LoRA ==========

print("===> LoRA ...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 6. TrainingArguments & Trainer ==========

print("===> TrainingArguments ...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,

    num_train_epochs=15,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    save_on_each_node=True,


)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_eval_ds,
    data_collator=data_collator,
)

# ========== 7. 开始训练 ==========

if __name__ == "__main__":
    trainer.train()
    eval_result = trainer.evaluate()
    print("Eval result:", eval_result)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
