from inference import run_inference
from analyze_results import load_results, print_grade_summary, find_reason_by_page

if __name__ == "__main__":
    run_inference()

    results = load_results()
    print_grade_summary(results)

    while True:
        user_input = input("\n🔍 이유를 보고 싶은 페이지 번호를 입력하세요 (0 입력 시 종료): ")
        if user_input.strip() == "0":
            break
        try:
            page = int(user_input)
            find_reason_by_page(results, page)
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
