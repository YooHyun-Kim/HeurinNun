import sys
import json
import os
from pathlib import Path

from module.classifier import classify_pdf_document
from module.text_table_parser import extract_tables_from_pdf
from module.sampling import structure_based_sampling
# 변경: ResNet/DenseNet 평가용 predict_image 대신 Tip-Adapter용 불러오기
from module.tip_adapter.tip_adapter import predict_image_tip_adapter
from module.llm.llm_main import llm_pipeline
from PIL import Image  # 이미지 로딩용

def save_jsonl(pages, output_path):
    seen = set()
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            key = (page["page"], page.get("text", ""), ",".join(page.get("image", [])))
            if key not in seen:
                seen.add(key)
                json.dump(page, f, ensure_ascii=False)
                f.write("\n")

def main():
    pdf_path    = "testdata/인사평가모음.pdf"
    output_dir  = Path("output")
    output_dir.mkdir(exist_ok=True)
    jsonl_path  = output_dir / "document.jsonl"

    # 1) 페이지별 텍스트·이미지 경로만 뽑기 (이미지 분류는 여기서 하지 않음)
    doc_type, doc_text, _, page_texts, page_images = classify_pdf_document(
        pdf_path, return_pages=True
    )

    # 2) 표 추출 및 샘플링
    all_tables              = extract_tables_from_pdf(pdf_path)
    sampled_page_numbers, _ = structure_based_sampling(pdf_path, num_pages=40)
    print(f"\n📄 샘플링된 페이지: {sampled_page_numbers}")

    result_pages = []
    for page_num in sampled_page_numbers:
        text        = page_texts[page_num] if page_num < len(page_texts) else ""
        image_paths = page_images[page_num] if page_num < len(page_images) else []

        print(f"\n📄 Page {page_num + 1} 처리 중...")

        # 3) Tip-Adapter로 이미지 태그 예측
        image_preds = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                # 1) Tip-Adapter로 예측
                preds = predict_image_tip_adapter(img)  # ex) ['흐름도'] or ['기타']
                # 2) '기타' 태그만 걸러내고 남은 태그만 사용
                preds = [p for p in preds if p != "기타"]
                # 3) 남은 태그가 있을 때만 결과 리스트에 추가
                if preds:
                    image_preds.extend(preds)
                    print(f"🖼️ {img_path} → {preds}")
            except Exception as e:
                print(f"⚠️ 이미지 태그 예측 실패 ({img_path}): {e}")

        # 4) 텍스트 vs 표 출력 (원본 로직 유지)
        if doc_type in [1, 3, 5]:
            matched_tables = [df for t_page, df in all_tables if t_page == page_num]
            if matched_tables:
                for df in matched_tables:
                    print(f"\n📊 Page {page_num + 1} 표:")
                    print(df)
            else:
                print("✅ 일반 텍스트 출력:")
                print(text)

        # 5) 페이지 결과 조립
        entry = {
            "page": page_num + 1,
            "text": text,
            "image": image_preds
        }
        result_pages.append(entry)

    # 6) JSONL 저장
    save_jsonl(result_pages, jsonl_path)
    print(f"\n✅ JSONL 저장 완료: {jsonl_path}")

    # 7) LLM 파이프라인 호출
    result = llm_pipeline()
    print(f"\n✅ LLM 파이프라인 결과: {result}")
    return result
if __name__ == "__main__":
    main()
