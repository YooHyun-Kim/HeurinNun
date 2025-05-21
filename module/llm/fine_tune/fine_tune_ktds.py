from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch

# 1. 모델 ID 및 양자화 설정
model_id = "AIDX-ktds/ktdsbaseLM-v0.13-onbased-llama3.1"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 2. Tokenizer 및 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    attn_implementation="eager"
)

# 3. LoRA 구성 (LLaMA3 기반 target modules)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    return tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize)

# 5. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./checkpoints_ktds_llama3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none"
)

# 6. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 7. 학습 수행
trainer.train()

# 8. 최종 LoRA adapter 저장
model.save_pretrained("./checkpoints_ktds_llama3/finetuned_ktds_llama3_qlora", save_adapter=True)
tokenizer.save_pretrained("./checkpoints_ktds_llama3/finetuned_ktds_llama3_qlora")
