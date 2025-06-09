import pdfplumber
import re

def structure_based_sampling(pdf_path, num_pages=40):
    sampled_pages = []
    sampled_page_numbers = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        if total_pages <= num_pages:
            print(f"📄 PDF는 {total_pages}페이지입니다. 전체 페이지를 사용합니다.")
            sampled_indices = list(range(total_pages))
        else:
            page_scores = []

            for i, page in enumerate(pdf.pages):  # i: 0-based
                text = page.extract_text() or ""
                score = 0

                if re.search(r"목차|contents", text, re.IGNORECASE):
                    score += 5
                words = page.extract_words()
                if words and isinstance(words[0], dict) and words[0].get("size", 0) > 15:
                    score += 3
                if len(page.extract_tables()) > 0:
                    score += 4
                if len(page.images) > 0:
                    score += 4
                if re.search(r"결론|요약|summary|conclusion", text, re.IGNORECASE):
                    score += 5

                page_scores.append((i, score))

            # 점수 기준 정렬 (점수 내림차순, 인덱스 오름차순)
            page_scores.sort(key=lambda x: (-x[1], x[0]))
            sampled_indices = [x[0] for x in page_scores[:num_pages]]

        # 🔥 핵심 수정: 직접 인덱스로 접근 (i-1 절대 X)
        sampled_pages = [pdf.pages[i].extract_text() or "" for i in sampled_indices]
        sampled_page_numbers = [i for i in sampled_indices]  # 1-based 번호 반환

    return sampled_page_numbers, sampled_pages
