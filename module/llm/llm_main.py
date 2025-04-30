from module.llm.inference import run_inference
from module.llm.analyze_results import load_results, print_grade_summary, find_reason_by_page
from module.llm.summarize_results import summarize_results

def llm_pipeline():
    # 1. 문서 분류 실행
    run_inference()

    # 2. 결과 로드 및 요약 출력
    results = load_results()
    print_grade_summary(results)
    summarize_results()
    # 3. 페이지별 이유 조회 루프
    while True:
        user_input = input("\n🔍 이유를 보고 싶은 페이지 번호를 입력하세요 (0 입력 시 종료): ")
        if user_input.strip() == "0":
            print("👋 종료합니다.")
            break
        try:
            page = int(user_input)
            find_reason_by_page(results, page)
        except ValueError:
            print("❌ 숫자를 입력해주세요.")

if __name__ == "__main__":
    llm_pipeline()
