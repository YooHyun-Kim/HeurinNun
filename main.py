import sys
import json
import os
from pathlib import Path
from module.classifier import classify_pdf_document
from module.text_table_parser import extract_tables_from_pdf
from module.sampling import structure_based_sampling
from module.predict_image_class import predict_image
from module.llm.llm_main import llm_pipeline
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
    pdf_path = "data/simulation_data.pdf"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    jsonl_path = output_dir / "document.jsonl"

    doc_type, doc_text, image_classes, page_texts, page_images = classify_pdf_document(
    pdf_path, return_pages=True
    )

    all_tables = extract_tables_from_pdf(pdf_path)
    sampled_page_numbers, _ = structure_based_sampling(pdf_path, num_pages=40)

    print(f"\n📄 샘플링된 페이지: {sampled_page_numbers}")
    result_pages = []

    for page_num in sampled_page_numbers:
        text = page_texts[page_num] if page_num < len(page_texts) else ""
        image_paths = page_images[page_num] if page_num < len(page_images) else []

        print(f"\n📄 Page {page_num} 처리 중...")

        # 이미지 예측 수행
        image_preds = []
        for img_path in image_paths:
            try:
                pred = predict_image(img_path)
                image_preds.append(pred)
                print(f"🖼️ {img_path} → {pred}")
            except Exception as e:
                print(f"⚠️ 예측 실패: {e}")

        # 표 출력
        if doc_type in [1, 3, 5]:
            matched_tables = [df for t_page, df in all_tables if t_page == page_num]
            if matched_tables:
                for df in matched_tables:
                    print(f"\n📊 Page {page_num + 1} 표:")
                    print(df)
            else:
                print("✅ 일반 텍스트 출력:")
                print(text)

        entry = {
            "page": page_num + 1,  # 실제 페이지 넘버
            "text": text,
            "image": image_preds
        }
        result_pages.append(entry)

    save_jsonl(result_pages, jsonl_path)
    print(f"\n✅ JSONL 저장 완료: {jsonl_path}")



if __name__ == "__main__":
    main()
    llm_pipeline()
