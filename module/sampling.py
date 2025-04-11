import pdfplumber
import re
import random

def structure_based_sampling(pdf_path, num_pages=40):
    sampled_pages = []
    total_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        # 페이지 수가 num_pages보다 적으면 전체 페이지를 그대로 사용
        if total_pages < num_pages:
            print(f"PDF 파일의 페이지 수가 {num_pages} 페이지 미만이므로 전체 페이지를 사용합니다.")
            sampled_page_numbers = list(range(total_pages))  # 전체 페이지 번호 사용
            sampled_pages = [pdf.pages[i-1].extract_text() or "" for i in sampled_page_numbers]
        else:
            page_scores = []  # 페이지별 중요도 점수 저장

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                score = 0

                # 샘플링 로직 (목차, 큰 글씨, 표/그래프 포함된 페이지 등)
                if re.search(r"목차|contents", text, re.IGNORECASE):
                    score += 5
                words = page.extract_words()
                if words and "size" in words[0] and words[0]["size"] > 15:
                    score += 3
                if len(page.extract_tables()) > 0:
                    score += 4
                if len(page.images) > 0:
                    score += 4
                if re.search(r"결론|요약|summary|conclusion", text, re.IGNORECASE):
                    score += 5

                page_scores.append((i+1, score, text))

            page_scores.sort(key=lambda x: x[1], reverse=True)  # 점수 높은 순으로 정렬
            sampled_page_numbers = [x[0] for x in page_scores[:num_pages]]
            sampled_pages = [pdf.pages[i-1].extract_text() or "" for i in sampled_page_numbers]

    return sampled_page_numbers, sampled_pages
