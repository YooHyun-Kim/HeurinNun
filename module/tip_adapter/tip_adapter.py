# module/tip_adapter/tip_adapter.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from .bank_retrieval import load_memory_bank, retrieve_tags

# CLIP & 메모리뱅크 초기화 (수정 없음)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
MEMORY_BANK_PATH = "memory_bank/memory_bank.pt"
keys, values, idx2label = load_memory_bank(MEMORY_BANK_PATH)

# 여기에 우리가 사용할 이미지 태그 이름을 모두 적어주세요
ALLOWED_TAGS = [
    "흐름도", "건축도면", "디바이스",
    "장비도면", "회로도면", "로고", "그래프"
]

def predict_image_tip_adapter(
    img: Image.Image,
    tau: float = 0.07,
    top_k: int = 5,
    threshold: float = 0.4
) -> list[str]:
    """
    Tip-Adapter 방식으로 이미지 분류:
    - 메모리뱅크에서 top_k 태그를 가져오고,
    - 그중 ALLOWED_TAGS에 속하면서 threshold 이상인 태그를 선택.
    - 없으면 ['기타'] 반환.
    """
    # 1) CLIP feature 추출
    inputs = processor(images=[img], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs).cpu()

    # 2) 메모리뱅크에서 태그 및 점수 가져오기
    tag_scores = retrieve_tags(feat, keys, values, idx2label,
                               tau=tau, top_k=top_k, threshold=0.0)
    # retrieve_tags에서 threshold=0.0으로 모든 점수를 받아옴

    # 3) 허용된 태그만 필터링
    filtered = [(label, score) for label, score in tag_scores if label in ALLOWED_TAGS]

    # 4) 점수 기준으로 출력 결정
    if filtered:
        # 점수 내림차순 정렬
        filtered.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = filtered[0]
        if best_score >= threshold:
            return [best_label]

    return ["기타"]
