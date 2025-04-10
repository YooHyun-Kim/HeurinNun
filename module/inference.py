from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import json

# 모델 로딩을 함수로 분리
def load_model(model_path="fine_tune/checkpoints/finetuned_gemma_qlora"):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    model.eval()
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

2급: 단순 설계도(예: 평면도), 내부 보고서(재정현황, 감사, 결산 등). 단독 설계도는 2급이나, 도메인 일치 시 1급으로 상향될 수 있음

3급: 1급,2급 요소가 하나도 없는 일반적인 문서(오픈소스, 공개 정보 등)

---

※ 판단 원칙:

- 하나의 문서에 여러 등급의 정보가 섞여 있을 경우, 가장 높은 등급을 기준으로 전체 보안등급을 판단합니다.
- 1급 요소가 단 하나라도 포함되어 있다면 해당 문서는 반드시 1급으로 분류합니다.
- 1급이 없고 2급과 3급 요소가 함께 포함된 경우에는 2급으로 분류합니다.
- 단일 등급 정보만 포함된 경우에는 해당 등급으로 분류합니다.

---

다음 문서의 내용을 분석하여 보안등급을 1급, 2급, 3급 중 하나로 판단하고, 그 이유도 간단히 설명하세요.

문서:
\"\"\"{doc_text}\"\"\"

보안등급 및 이유:
"""

# 메인 인퍼런스 함수
def run_inference(input_path="document.jsonl", output_path="output_results_jsonlver.jsonl", model_path="fine_tune/checkpoints/finetuned_gemma_qlora"):
    tokenizer, model = load_model(model_path)

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            data = json.loads(line)
            doc_text = data.get("text", "").strip()
            page_num = data.get("page", None)
            
            if not doc_text or page_num is None:
                print("⚠️ 입력 데이터에 'text' 또는 'page'가 누락되었습니다. 건너뜁니다.")
                continue

            prompt = base_prompt.format(doc_text=doc_text)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True
                )

            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if "문서:" in generated_text:
                generated_text = generated_text.split("문서:")[0].strip()

            grade, reason = parse_grade_and_reason(generated_text)

            result = {
                "page": page_num,
                "grade": grade,
                "reason": reason
            }
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"✅ {page_num}페이지 완료 - {grade}")
