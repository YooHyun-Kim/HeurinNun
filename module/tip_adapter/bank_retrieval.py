import os
import json
import torch
import torch.nn.functional as F
import fitz  # PyMuPDF
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO

def load_memory_bank(path: str):
    """메모리 뱅크(.pt)에서 keys, values, idx2label 로드"""
    data = torch.load(path)
    keys      = data["keys"]
    values    = data["values"]
    label2idx = data["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}
    return keys, values, idx2label

def extract_page_images(pdf_path: str, page_num: int):
    """PyMuPDF로 PDF 페이지에서 내장 이미지를 모두 추출하여 PIL 이미지 리스트 반환"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    images = []
    for img_meta in page.get_images(full=True):
        xref = img_meta[0]
        pix = fitz.Pixmap(doc, xref)
        try:
            img_bytes = pix.tobytes("png")
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Page {page_num} xref {xref} 파싱 실패: {e}")
        finally:
            pix = None
    return images

def retrieve_tags(q, keys, values, idx2label, tau=0.07, top_k=2, threshold=0.2):
    """쿼리 q와 메모리 키 간 유사도 계산 후 top_k 태그 반환 (threshold 미만은 '기타')"""
    sims    = F.cosine_similarity(q, keys)        # [N]
    weights = F.softmax(sims / tau, dim=0)        # [N]
    scores  = torch.zeros(len(idx2label))
    for w, idx in zip(weights, values):
        scores[idx.item()] += w.item()
    max_score = scores.max().item()
    if max_score < threshold:
        return [("기타", max_score)]
    topk = torch.topk(scores, k=top_k)
    return [(idx2label[i.item()], float(scores[i])) for i in topk.indices]

def process_pdf_images_sequential(pdf_path: str,
                                  memory_bank_path: str = "memory_bank/memory_bank.pt",
                                  tau: float = 0.07,
                                  top_k: int = 2,
                                  threshold: float = 0.2):
    """
    각 페이지내 모든 이미지에 대해 순차적으로 JSONL에 기록:
    {"page": <page_num>, "image": <top_tags>}
    """
    keys, values, idx2label = load_memory_bank(memory_bank_path)
    device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor               = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model                   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") \
                                 .to(device).eval()

    output_path = "document.jsonl"
    # 파일 열고 한 줄씩 기록
    with open(output_path, "w", encoding="utf-8") as fout:
        doc = fitz.open(pdf_path)
        for page_num in range(1, len(doc) + 1):
            images = extract_page_images(pdf_path, page_num)
            for img in images:
                inputs = processor(images=[img], return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    q_img = model.get_image_features(**inputs).cpu()
                top_tags = retrieve_tags(q_img, keys, values, idx2label,
                                         tau=tau, top_k=top_k, threshold=threshold)
                entry = {
                    "page":  page_num,
                    "image": top_tags
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Sequential image-level tags saved to {output_path}")

if __name__ == "__main__":
    pdf_file = "data/pretraining_clip/index_10.pdf"
    process_pdf_images_sequential(pdf_file)
