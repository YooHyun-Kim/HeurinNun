from transformers import AutoTokenizer, AutoModelForCausalLM
from classifier import classify_pdf_document

##문서 경로 설정
pdf_path = "data/sample_Data_idx_4_only_text.pdf"
doc_type, doc_text = classify_pdf_document(pdf_path)

# 프롬프트 규칙 정의 (문서 내용은 나중에 붙임)
base_prompt = """
다음은 기술 분야의 보안등급 분류 기준입니다:

1급: 기술정보(연구, 부품, 설계도, 장비, 디자인 요소), 개인정보, 거래 납품 리스트(예: 제조 기술 명시 등), 특정 단어(기밀유지, 보안관리 등)가 포함된 민감한 문서

2급: 단순 설계도(예: 평면도), 내부 보고서(재정현황, 감사, 결산 등). 단독 설계도는 2급이나, 도메인 일치 시 1급으로 상향될 수 있음

3급: 일반적인 문서(오픈소스, 공개 정보 등)

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


# 전체 프롬프트 구성
prompt = base_prompt.format(doc_text=doc_text)

# 모델 ID 설정
model_id = "recoilme/recoilme-gemma-2-9B-v0.4"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype="auto"
)

# 입력 토크나이즈
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]

# 텍스트 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# 프롬프트 이후 생성된 토큰만 추출
generated_tokens = outputs[0][input_length:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# 결과 출력
print("문서 유형:", doc_type)
print(generated_text.strip())
