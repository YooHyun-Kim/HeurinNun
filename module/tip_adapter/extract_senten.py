## 이 코드는, 문서에서 문장 단위로, 텍스트를 저장 한 뒤에, json 파일로 저장함.
## 이 과정이 필요한 이유는 tip_adapter를 사용하기 위해서 , clip의 text encoder에는 토큰 수 제한이 있기에 페이지 내 모든 문장을 한번에 처리 할 수 없기 때문이다.


import os
import pdfplumber
import re
import json

def extract_sentences_from_pdf(pdf_path: str, output_dir: str = "senten_output"):
    """
    PDF 파일에서 페이지별로 텍스트를 추출하고,
    문장 단위로 분리하여 지정된 폴더에 JSON으로 저장합니다.
    """
    # PDF 파일명에서 확장자 제거 후 기본 이름 추출
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, f"{base_name}_page_sentences.json")

    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            sentences = re.split(r'(?<=[\.\?\!])\s+', text)
            pages_data.append({
                "page": page_number,
                "sentences": sentences
            })

    # JSON으로 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)

    print(f"PDF '{pdf_path}'의 {len(pages_data)}페이지에서 문장 단위 추출 완료.")
    print(f"결과가 '{output_json}'에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    pdf_file = "data/pretraining_clip/index_10.pdf"  # 실제 PDF 경로
    extract_sentences_from_pdf(pdf_file)

