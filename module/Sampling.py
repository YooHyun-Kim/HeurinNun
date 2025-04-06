## 문서구조 기반 샘플링 Ver.04 (점수정렬+점수정렬제거)

import pdfplumber
import re
import random

def structure_based_sampling(pdf_path, num_pages=40):
    sampled_pages = []
    total_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # 페이지 수가 40 미만이면 샘플링 중단
        if total_pages < 40:
            print("PDF 파일의 페이지 수가 40페이지 미만이므로 샘플링하지 않습니다.")
            return [], []
        
        page_scores = []  # 페이지별 중요도 점수 저장
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            score = 0
            
            # 목차가 있는 페이지
            if re.search(r"목차|contents", text, re.IGNORECASE):
                score += 5

            # 제목이 큰 글씨 (첫 번째 단어 폰트 크기 분석)
            words = page.extract_words()
            if words and "size" in words[0] and words[0]["size"] > 15:
                score += 3
            
            # 표/그래프가 포함된 페이지
            if len(page.extract_tables()) > 0:
                score += 4
                
            if len(page.images) > 0:
                score += 4

            # 결론/요약이 포함된 페이지
            if re.search(r"결론|요약|summary|conclusion", text, re.IGNORECASE):
                score += 5
            
            # 페이지와 점수를 저장
            page_scores.append((i+1, score, text))
        
        # 페이지 수가 500 미만일 경우 점수 정렬을 적용
        if total_pages < 500:
            # 점수 정렬이 있는 경우
            page_scores.sort(key=lambda x: x[1], reverse=True)
        
            # 중요도가 높은 페이지를 선택
            sampled_pages = []
            sampled_page_numbers = []
            
            unique_scores = list(set(score for _, score, _ in page_scores))
            unique_scores.sort(reverse=True)  # 점수 내림차순 정렬
            
            for score in unique_scores:
                pages_with_same_score = [(p, t) for p, s, t in page_scores if s == score]
                random.shuffle(pages_with_same_score)  # 같은 점수 내에서 랜덤 섞기
                
                for page_num, text in pages_with_same_score:
                    if len(sampled_pages) < num_pages and page_num not in sampled_page_numbers:
                        sampled_page_numbers.append(page_num)
                        sampled_pages.append(text)
                    if len(sampled_pages) >= num_pages:
                        break
                if len(sampled_pages) >= num_pages:
                    break
        else:
            # 500페이지 이상일 경우 점수 정렬 없이 랜덤 샘플링
            sampled_page_numbers = random.sample(range(1, total_pages + 1), num_pages)
            sampled_pages = [pdf.pages[i-1].extract_text() or "" for i in sampled_page_numbers]
        
    return sampled_page_numbers, sampled_pages

pdf_path = r"C:\Users\LG\Desktop\하이브리드굴삭기시스템설계기술개발.pdf"
sampled_page_numbers, sampled_texts = structure_based_sampling(pdf_path, num_pages=40)
print(sampled_page_numbers)
# 결과 출력
if sampled_page_numbers:
    for i, (page_num, text) in enumerate(zip(sampled_page_numbers, sampled_texts[:7])):  # 앞부분만 출력
        print(f"\n 원본 PDF의 {page_num} 페이지:\n{text[:500]}...\n")
