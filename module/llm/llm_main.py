from module.llm.inference import run_inference
from module.llm.analyze_results import load_results, print_grade_summary, find_reason_by_page
from module.llm.summarize_results import summarize_results
import time
from datetime import datetime
def llm_pipeline():
    # 1. 문서 분류 실행
    # 시간 측정 시작
    start_time = time.time()
    print(f"🕒 추론 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    
    run_inference(model_path="module/llm/fine_tune/checkpoints/finetuned_gemma_qlora")  # 젬마모델 사용 - 디폴트
    #run_inference(model_path="module/llm/fine_tune/checkpoints_ktds_llama3/finetuned_ktds_llama3_qlora")  # aidx모델 사용
    #run_inference(model_path="module/llm/fine_tune/checkpoints_llama3/finetuned_llama3_qlora") # seokdong모델 사용


    # 2. 결과 로드 및 요약 출력
    results = load_results()
    page_result = print_grade_summary(results)
    result = summarize_results()

# 추론 종료 시간 출력
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)

    print(f"✅ 추론 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱ 총 추론 시간: {elapsed_min}분 {elapsed_sec}초")

    #3. 페이지별 이유 조회 루프
    while True:
        user_input = input("\n 이유를 보고 싶은 페이지 번호를 입력하세요 (0 입력 시 종료): ")
        if user_input.strip() == "0":
            print(" 종료합니다.")
            break
        try:
            page = int(user_input)
            find_reason_by_page(results, page)
        except ValueError:
            print(" 숫자를 입력해주세요.")
    return page_result,result
if __name__ == "__main__":
    llm_pipeline()
