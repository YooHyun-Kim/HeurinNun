import json
import random
import gc
from collections import defaultdict, Counter

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer


def select_representative_reasons(reasons, top_k=2, random_k=1):
    cleaned = list({r.strip() for r in reasons if len(r.strip()) > 5})
    if len(cleaned) <= (top_k + random_k):
        return cleaned

    sorted_by_length = sorted(cleaned, key=len, reverse=True)
    top_reasons = sorted_by_length[:top_k]
    remaining = list(set(cleaned) - set(top_reasons))
    random_reasons = random.sample(remaining, min(random_k, len(remaining)))

    return top_reasons + random_reasons


def extract_keywords_and_tfidf(reasons, stopwords=None):
    okt = Okt()
    noun_docs = []
    all_nouns = []

    for reason in reasons:
        nouns = okt.nouns(reason)
        if stopwords:
            nouns = [n for n in nouns if n not in stopwords]
        noun_docs.append(" ".join(nouns))
        all_nouns.extend(nouns)

    if len(reasons) > 1:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(noun_docs)
        tfidf_scores = tfidf_matrix.toarray().sum(axis=0)
        tfidf_vocab = vectorizer.get_feature_names_out()
        tfidf_dict = {word: score for word, score in zip(tfidf_vocab, tfidf_scores)}
        top_tfidf_words = [word for word, _ in sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:5]]
    else:
        top_tfidf_words = [word for word, _ in Counter(all_nouns).most_common(5)]

    return top_tfidf_words


def summarize_results(results_path="output/output_results.jsonl"):
    grade_reason_map = defaultdict(list)

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            grade = data.get("grade", "").strip()
            reason = data.get("reason", "").strip()
            if grade and reason:
                grade_reason_map[grade].append(reason)

    for level in ["1급", "2급", "3급"]:
        if grade_reason_map[level]:
            final_grade = level
            reasons = [r.replace("이유:", "").strip() for r in grade_reason_map[level]]
            break
    else:
        print("❌ No valid security grade reason found.")
        return None

    stopwords = {'함유', '정보', '결과', '기반', '포함', '관련', '내용', '실제','문헌','다수','보이','중요','대한','등','것','임',"포", "함",'액',"사","해당","외","또한","검색","판단","모든","다른","사람","동일","나타남","확인","사용"}

    top_keywords = extract_keywords_and_tfidf(reasons, stopwords)
    selected = select_representative_reasons(reasons)
    
    # print(f"\n✅ Final Security Grade: {final_grade}")
    # print("📌 Representative Reasons:")
    # for i, r in enumerate(selected, 1):
    #     print(f"{i}. {r}")

    # print("\n📌 Top 5 Keywords from Reasons:")
    # for idx, word in enumerate(top_keywords, 1):
    #     print(f"keyword{idx}: {word}")

    gc.collect()
    
    return {
    "grade": final_grade,
    "reason": ",\u00A0".join(selected),  # 👉 이유는 쉼표+공백으로 연결
    "keyword": top_keywords  # 👉 리스트 형태로 반환
    }


if __name__ == "__main__":
    result = summarize_results("../../output/output_results.jsonl")
    print(result)
