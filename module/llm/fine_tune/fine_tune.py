# finetune_lora/finetune.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch

model_id = "recoilme/recoilme-gemma-2-9B-v0.4"

# 1. Quantization 설정 (QLoRA용)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # 또는 "fp4"
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Tokenizer & 4bit Model 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    attn_implementation="eager"
)

# 3. LoRA 설정
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 4. 데이터셋 로드
dataset = load_dataset("json", data_files={"train": "data/train_reformatted.jsonl"})["train"]

def tokenize(example):
    full_prompt = f"{example['prompt']} {example['response']}"
    return tokenizer(full_prompt, truncation=True, padding="max_length", max_length=2048)

tokenized_dataset = dataset.map(tokenize)

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none",  # wandb 끄기
)

# 6. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 7. 학습 시작
trainer.train()

# 8. 저장
# adapter만 저장 (정확한 방식)
model.save_pretrained("./checkpoints/finetuned_gemma_qlora", save_adapter=True)

tokenizer.save_pretrained("./checkpoints/finetuned_gemma_qlora")
