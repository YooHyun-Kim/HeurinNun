import json

# 기준 설명 프롬프트 템플릿
base_prompt = """다음은 기술 분야의 보안등급 분류 기준입니다:

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

보안등급 및 이유:"""

# 파일 경로 설정
input_file = "data/train.jsonl"
output_file = "data/train_reformatted.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        data = json.loads(line)
        # 기존 짧은 프롬프트에서 문서 텍스트 추출
        if data["prompt"].startswith("문서:"):
            doc_text = data["prompt"].split("문서:")[1].split("\n")[0].strip()
        else:
            doc_text = data["prompt"].strip()
        # 새로운 포맷으로 구성
        new_prompt = base_prompt.format(doc_text=doc_text)
        new_data = {
            "prompt": new_prompt,
            "response": data["response"]
        }
        f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print(f"✅ 변환 완료! 저장 경로: {output_file}")
