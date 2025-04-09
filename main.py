import sys
from module.classifier import classify_pdf_document
from module.text_table_parser import extract_tables_from_pdf

def main():
    pdf_path = "data/sample_Data_idx_6_text_image(text).pdf"  # 테스트용 PDF 경로

    # 1. 문서 분류 및 정보 추출
    doc_type, doc_text, image_features = classify_pdf_document(pdf_path)
    print(f"\n📄 문서 유형: {doc_type}")

    # 2. 유형별 처리 분기
    if doc_type in [1, 3, 5]:
        print("\n📊 텍스트 기반 표 추출 시도:")
        tables = extract_tables_from_pdf(pdf_path)

        if tables:
            for page_num, df in tables:
                print(f"\n📄 Page {page_num + 1} 표:")
                print(df)

            # ✅ 문서 유형 5는 텍스트도 함께 출력
            if doc_type == 5:
                print("\n📝 문서 전체 텍스트:")
                print(doc_text)

        else:
            print("\n✅ 표는 감지되지 않음 → 일반 텍스트 출력:")
            print(doc_text)

    elif doc_type == 2:
        print("\n🖼️ 이미지만 포함된 문서 (텍스트 없음)")
        print(f"이미지 개수: {len(image_features)}")
        # TODO: 이미지 feature → classifier로 넘겨 클래스 추론

    elif doc_type == 4:
        print("\n🖼️ 이미지 내 텍스트(OCR)만 존재하는 문서")
        print(doc_text)
        print("OCR 기반 텍스트가 존재함 (본문 텍스트는 없음)")
        # TODO: OCR 텍스트 따로 저장할지 결정

    else:
        print("\n⚠️ 미분류 문서입니다.")

    # 3. (미래 작업) → LLM 연동을 위한 텍스트 조합 및 전달
    # TODO:
    # - doc_text + 표 데이터 + 이미지 분류 결과 종합
    # - 프롬프트 구성 후 LLM 호출

if __name__ == "__main__":
    main()
