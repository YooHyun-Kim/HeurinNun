import sys
import json
import os
from pathlib import Path
from module.classifier import classify_pdf_document
from module.text_table_parser import extract_tables_from_pdf
from module.sampling import structure_based_sampling
from module.predict_image_class import predict_image

def save_jsonl(pages, output_path):
    seen = set()
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            key = (page["page"], page.get("text", ""), page.get("image", ""))
            if key not in seen:
                seen.add(key)
                json.dump(page, f, ensure_ascii=False)
                f.write("\n")

def main():
    pdf_path = "data/simulation_data.pdf"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    jsonl_path = output_dir / "document.jsonl"

    # 전체 문서 분류 및 정보 추출 (한 번만 수행)
    doc_type, doc_text, image_features, page_texts, page_images = classify_pdf_document(
        pdf_path, return_pages=True
    )

    # 전체 테이블도 한 번만 추출
    all_tables = extract_tables_from_pdf(pdf_path)  # [(page_num, df), ...]

    # 문서 샘플링
    sampled_page_numbers, _ = structure_based_sampling(pdf_path, num_pages=40)
    print(f"\n📄 샘플링된 페이지: {sampled_page_numbers}")

    result_pages = []

    for page_num in sampled_page_numbers:
        text = page_texts[page_num] if page_num < len(page_texts) else ""
        image_class = ""

        print(f"\n📄 Page {page_num} 처리 중...")

        # 1. 텍스트 기반 문서
        if doc_type in [1, 3, 5]:
            print(f"📊 텍스트 기반 표 추출 시도: {page_num}")
            matched_tables = [df for t_page, df in all_tables if t_page == page_num]
            if matched_tables:
                for df in matched_tables:
                    print(f"\n📄 Page {page_num + 1} 표:")
                    print(df)
            else:
                print("✅ 표는 감지되지 않음 → 일반 텍스트 출력:")
                print(text)

        # 2. 이미지 예측 (텍스트가 없거나 이미지 기반인 경우)
        if page_num < len(page_images):
            img_path = page_images[page_num]
            if os.path.exists(img_path):
                try:
                    image_class = predict_image(img_path)
                    print(f"🖼️ 이미지 예측 결과: {image_class}")
                except Exception as e:
                    print(f"⚠️ 이미지 예측 실패: {e}")

        # 3. OCR 기반 문서
        if doc_type == 4:
            print(f"🖼️ OCR 기반 텍스트 (본문 없음): {text}")

        # 최종 entry 생성
        entry = {
            "page": page_num,
            "text": text,
            "image": image_class
        }
        result_pages.append(entry)

    # 중복 제거하여 저장
    save_jsonl(result_pages, jsonl_path)
    print(f"\n✅ JSONL 저장 완료: {jsonl_path}")

if __name__ == "__main__":
    main()
