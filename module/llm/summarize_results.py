import torch
import json
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_base_model():
    base_model_id = "beomi/open-llama-2-ko-7b"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        local_files_only=False   # 처음 다운로드할 때만 False
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,     # float16 사용
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=False         # 처음 다운로드할 때만 False
    )

    model.eval()
    return tokenizer, model

# 📌 요약 프롬프트 템플릿
summarize_prompt = """
다음은 각 페이지별로 보안등급을 판단한 이유 모음입니다. 
등급을 결정한 핵심적인 이유만 간단히 정리해서 작성하세요.
불필요한 반복은 피하고, 요약은 1~2문장 이내로 작성하세요.

불필요한 설명, 기관명, 날짜 등은 생략하고, 핵심 기술/설계/구성 정보만 강조해 주세요.

이유 목록:
\"\"\"{reasons}\"\"\"

요약:
"""

# ✅ 요약 수행 함수
def summarize_results(results_path="output/output_results.jsonl"):
    # 1. 모델 로드
    tokenizer, model = load_base_model()

    # 2. 등급 및 이유 수집 (등급별로 분리 저장)
    grade_reason_map = {"1급": [], "2급": [], "3급": []}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            grade = data.get("grade", "").strip()
            reason = data.get("reason", "").strip()
            if grade and reason and grade in grade_reason_map:
                grade_reason_map[grade].append(reason)

    # 3. 최종 등급 결정 및 해당 이유 선택
    if grade_reason_map["1급"]:
        final_grade = "1급"
        selected_reasons = grade_reason_map["1급"]
    elif grade_reason_map["2급"]:
        final_grade = "2급"
        selected_reasons = grade_reason_map["2급"]
    else:
        final_grade = "3급"
        selected_reasons = grade_reason_map["3급"]


    # 4. 프롬프트 구성
    prompt = summarize_prompt.format(reasons="\n".join(selected_reasons))

    # 5. 입력 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    for k in inputs:
        inputs[k] = inputs[k].to(model.device) if k == "input_ids" else inputs[k].to(model.device).to(torch.float16)
    input_length = inputs["input_ids"].shape[1]

    # 6. 요약 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2
        )

    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 7. "요약:" 제거 및 마무리
    if "요약:" in generated_text:
        generated_text = generated_text.split("요약:")[-1].strip()
    summary = generated_text.strip()

    # 8. 결과 출력
    print(f"✅ 최종 보안등급: {final_grade}")
    print(f"📝 요약 결과:\n{summary}")

    # 9. 메모리 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # # 10. JSON 반환
    # return {
    #     "final_grade": final_grade,
    #     "final_summary": summary
    # }
