from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import re
import json
from pathlib import Path


def load_model_gemma(model_path):
    model_path = Path(model_path).resolve()
    base_model_id = "recoilme/recoilme-gemma-2-9B-v0.4"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)

    # base model 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=True
    )

    # PEFT 양자화 모델용 준비
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA 구조 정의
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)

    # 학습된 adapter 로드
    model.load_adapter(str(model_path), adapter_name="default")

    model = model.half()
    model.eval()
    model.print_trainable_parameters()
    return tokenizer, model

def load_model_seokdong(model_path):
    model_path = Path(model_path).resolve()
    base_model_id = "SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx"  # 또는 다른 LLaMA3 모델

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=True
    )

    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, peft_config)
    model.load_adapter(str(model_path), adapter_name="default")

    model = model.half()
    model.eval()
    model.print_trainable_parameters()
    return tokenizer, model

def load_model_aidx(model_path):
    model_path = Path(model_path).resolve()
    base_model_id = "AIDX-ktds/ktdsbaseLM-v0.13-onbased-llama3.1"

    # 1. QLoRA 양자화 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Tokenizer 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Base 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=True
    )

    # 4. QLoRA 학습 준비
    base_model = prepare_model_for_kbit_training(base_model)

    # 5. LoRA 설정 (LLaMA3 전용)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. LoRA 어댑터 적용
    model = get_peft_model(base_model, peft_config)
    model.load_adapter(str(model_path), adapter_name="default")

    # 7. 평가모드 전환 및 반정밀도 설정
    model = model.half()
    model.eval()
    model.print_trainable_parameters()

    return tokenizer, model


# 등급과 이유 파싱 함수
def parse_grade_and_reason(text):
    match = re.search(r"(1급|2급|3급)", text)
    if match:
        grade = match.group(1)
        reason = text[match.end():].strip(" .:\n")
        reason = re.sub(r'^[\s,.:;\-]+', '', reason)
        reason = re.sub(r'[\s,.:;"\'\-\n]+$', '', reason)
    else:
        grade = "미상"
        reason = text.strip()
    return grade, reason

# 프롬프트 정의 템플릿
base_prompt = """
다음은 기술 분야의 보안등급 분류 기준입니다:

1급: 기술정보(연구, 부품, 설계도, 장비, 디자인 요소), 개인정보, 거래 납품 리스트(예: 제조 기술 명시 등), 특정 단어(기밀유지, 보안관리 등)가 포함된 민감한 문서

2급: 재정현황, 감사, 결산, 인사평가, 업부보고서, 징계문서와 같은 내부 보고서  

3급: 1급,2급 요소가 하나도 없는 일반적인 문서(오픈소스, 공개 정보 등)

---

※ 판단 원칙:

- 하나의 문서에 여러 등급의 정보가 섞여 있을 경우, 가장 높은 등급을 기준으로 전체 보안등급을 판단합니다.
- 1급 요소가 단 하나라도 포함되어 있다면 해당 문서는 반드시 1급으로 분류합니다.
- 1급이 없고 2급과 3급 요소가 함께 포함된 경우에는 2급으로 분류합니다.
- 단일 등급 정보만 포함된 경우에는 해당 등급으로 분류합니다.
- 아래에 제시된 "문서 내 포함 이미지 내용" 외에는 별도 이미지가 없으며, 해당 항목만 시각 정보로 간주합니다.

---

다음 문서의 내용을 분석하여 보안등급을 1급, 2급, 3급 중 하나로 판단하고, 그 이유도 간단히 설명하세요.

문서:
\"\"\"{doc_text}\"\"\"
문서 내 포함 이미지 내용 (해당 페이지의 이미지에서 추출된 시각 정보입니다):
\"\"\"{img_text}\"\"\"

보안등급 및 이유:
"""

# 메인 인퍼런스 함수
def run_inference(input_path="output/document.jsonl", output_path="output/output_results.jsonl", model_path="module/llm/fine_tune/checkpoints/finetuned_gemma_qlora"):
    if model_path == "module/llm/fine_tune/checkpoints/finetuned_gemma_qlora":
        tokenizer, model = load_model_gemma(model_path)  # 젬마모델 사용 - 디폴트
    elif model_path == "module/llm/fine_tune/checkpoints_ktds_llama3/finetuned_ktds_llama3_qlora":
        tokenizer, model = load_model_aidx(model_path)
    #tokenizer, model = load_model_aidx(model_path) # aidx모델 사용
    elif model_path == "module/llm/fine_tune/checkpoints_llama3/finetuned_llama3_qlora":
        tokenizer, model = load_model_seokdong(model_path)
        
    #tokenizer, model = load_model_seokdong(model_path) # seokdong모델 사용
    else:
        print("❌ 잘못된 모델 경로입니다. 젬마 모델을 사용합니다.")
        tokenizer, model = load_model_gemma(model_path)

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            data = json.loads(line)
            doc_text = data.get("text", "").strip()
            page_num = data.get("page", None)

            img_text_raw = data.get("image", "")
            img_text = ", ".join(img_text_raw) if isinstance(img_text_raw, list) else str(img_text_raw)
            img_text = img_text.strip()

            if not doc_text or page_num is None:
                print("⚠️ 입력 데이터에 'text' 또는 'page'가 누락되었습니다. 건너뜁니다.")
                continue

            prompt = base_prompt.format(doc_text=doc_text, img_text=img_text)

            # 1. Tokenize
            inputs = tokenizer(prompt, return_tensors="pt",truncation=True, max_length=2048)

            # 2. input_ids는 long 유지, attention_mask 등은 float16 변환
            for k in inputs:
                if k == "input_ids":
                    inputs[k] = inputs[k].to(model.device)
                else:
                    inputs[k] = inputs[k].to(model.device).to(torch.float16)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.0,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=False
                )

            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if "문서:" in generated_text:
                generated_text = generated_text.split("문서:")[0].strip()
            # 개행(\n) 기준으로 첫 문장만 추출
            generated_text = generated_text.split('\n')[0].strip()
            grade, reason = parse_grade_and_reason(generated_text)

            result = {
                "page": page_num,
                "grade": grade,
                "reason": reason
            }
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"✅ {page_num}페이지 완료 - {grade}")
    # 🎯 모델 제거 및 메모리 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()    
