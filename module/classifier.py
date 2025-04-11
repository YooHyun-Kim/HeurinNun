import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import os

from image_classifier.resnet import build_resnet, preprocess_image, get_image_features
import torch

# 모델 불러오기
model = build_resnet(feature_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 클래스 이름
class_names = ["흐름도", "건축도면", "디바이스", "장비도면", "회로도면"]

def classify_pdf_document(pdf_path, return_pages=False):
    doc = fitz.open(pdf_path)

    has_text = False
    has_image = False
    image_has_text = False
    full_text = ""

    page_texts = []      # 각 페이지 텍스트
    page_images = []     # 각 페이지의 이미지 경로 리스트
    image_classes = []   # 각 페이지의 예측 클래스 리스트

    for page_num, page in enumerate(doc):
        # 텍스트 추출
        text = page.get_text()
        if text.strip():
            has_text = True
            full_text += text + "\n"
        page_texts.append(text.strip())

        # 이미지 추출
        images = page.get_images(full=True)
        page_img_paths = []
        page_img_preds = []

        if images:
            has_image = True
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))

                # 이미지 저장
                img_path = f"output/page_{page_num + 1}_img_{img_idx + 1}.png"
                img_pil.save(img_path)
                page_img_paths.append(img_path)

                # OCR로 텍스트 유무 확인
                ocr_result = pytesseract.image_to_string(img_pil)
                if ocr_result.strip():
                    image_has_text = True

                # 이미지 분류
                try:
                    img_tensor = preprocess_image(img_pil)
                    image_feature = get_image_features(img_tensor, model, device)
                    class_idx = image_feature.argmax().item()
                    page_img_preds.append(class_names[class_idx])
                except Exception as e:
                    print(f"⚠️ 이미지 분류 실패 (page {page_num+1}, img {img_idx+1}): {e}")
                    page_img_preds.append("UNKNOWN")

        page_images.append(page_img_paths)
        image_classes.append(page_img_preds)

    # 문서 유형 판별
    doc_type = 0
    if has_text and not has_image:
        doc_type = 1
    elif has_image and not has_text and not image_has_text:
        doc_type = 2
    elif has_text and has_image and not image_has_text:
        doc_type = 3
    elif has_image and not has_text and image_has_text:
        doc_type = 4
    elif has_text and has_image and image_has_text:
        doc_type = 5

    if return_pages:
        return doc_type, full_text.strip(), image_classes, page_texts, page_images
    else:
        return doc_type, full_text.strip(), image_classes
