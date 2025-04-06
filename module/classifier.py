import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_classifier.resnet import build_resnet
from image_classifier.resnet import preprocess_image, get_image_features
import torch

def classify_pdf_document(pdf_path, device="cpu"):
    doc = fitz.open(pdf_path)

    has_text = False
    has_image = False
    image_has_text = False
    full_text = ""
    image_features = []

    # 모델 불러오기 (ResNet50 feature extractor)
    model = build_resnet(feature_only=True)
    model.to(device)
    model.eval()

    for page in doc:
        # 1. 텍스트 추출
        text = page.get_text()
        if text.strip():
            has_text = True
            full_text += text + "\n"

        # 2. 이미지 + OCR + Feature 추출
        images = page.get_images(full=True)
        if images:
            has_image = True
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))

                # OCR: 이미지 내부 텍스트 검출
                ocr_result = pytesseract.image_to_string(img_pil)
                if ocr_result.strip():
                    image_has_text = True
                    if not has_text:
                        full_text += ocr_result + "\n"  # ← OCR 텍스트도 저장
                    break

                # 이미지 feature 추출
                img_tensor = preprocess_image(img_pil)  # 이건 이미지 객체로 받을 수 있도록 수정할게!
                feature = get_image_features(img_tensor, model, device=device)
                image_features.append(feature)

    # 분류
    if has_text and not has_image: ## 텍스트만
        doc_type = 1
    elif has_image and not has_text and not image_has_text: ## 이미지만
        doc_type = 2
    elif has_text and has_image and not image_has_text: ## 텍스트 + 이미지
        doc_type = 3
    elif has_image and not has_text and image_has_text: ## 이미지(+텍스트)
        doc_type = 4
    elif has_text and has_image and image_has_text: ## 텍스트 + 이미지(+텍스트)
        doc_type = 5
    else:
        doc_type = 0

    return doc_type, full_text.strip(), image_features
