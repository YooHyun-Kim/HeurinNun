import json
from collections import Counter

def load_results(file_path="output_results_jsonlver.jsonl"):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def determine_overall_grade(results):
    grades = [r["grade"] for r in results]
    if "1급" in grades:
        return "1급"
    elif "2급" in grades:
        return "2급"
    else:
        return "3급"

def print_grade_summary(results):
    page_by_grade = {"1급": [], "2급": [], "3급": []}
    for r in results:
        grade = r["grade"]
        page = r["page"]
        page_by_grade[grade].append(page)

    print("\n📘 전체 보안등급 판단 결과")
    overall = determine_overall_grade(results)
    print(f"📌 최종 등급: {overall}")

    for grade in ["1급", "2급", "3급"]:
        pages = page_by_grade[grade]
        if pages:
            page_list = ", ".join(str(p) for p in sorted(pages))
            print(f"🔹 {grade} 페이지: {page_list}")

def find_reason_by_page(results, page_number):
    for r in results:
        if r["page"] == page_number:
            print(f"\n📄 페이지 {page_number}")
            print(f"등급: {r['grade']}")
            print(f"이유: {r['reason']}")
            return
    print(f"\n⚠️ 페이지 {page_number}에 대한 정보가 없습니다.")


