import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_classifier.resnet import build_resnet, preprocess_image, get_image_features
import torch

# 이미지를 분류할 ResNet 모델 로드
model = build_resnet(feature_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ResNet을 사용할 때, 각 클래스에 해당하는 텍스트 (예: '건축도면', '회로도면' 등)
class_names = ["흐름도", "건축도면", "디바이스", "장비도면", "회로도면"]

def classify_pdf_document(pdf_path, return_pages=False):
    doc = fitz.open(pdf_path)
    
    has_text = False
    has_image = False
    full_text = ""
    image_has_text = False
    page_texts = []  # 페이지별 텍스트 저장용 리스트
    page_images = []  # 페이지별 이미지 경로 저장용 리스트
    image_classes = []  # 페이지별 이미지 클래스 (텍스트) 저장용 리스트

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            has_text = True
            full_text += text + "\n"
            page_texts.append(text.strip())  # 각 페이지 텍스트 추가

        images = page.get_images(full=True)
        if images:
            has_image = True
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))
                
                image_path = f"output/page_{page_num+1}.png"
                img_pil.save(image_path)
                page_images.append(image_path)  # 각 페이지 이미지 경로 추가

                # 이미지 내 텍스트 확인 (OCR)
                ocr_result = pytesseract.image_to_string(img_pil)
                if ocr_result.strip():
                    image_has_text = True

                # 이미지 feature 추출 및 분류
                img_tensor = preprocess_image(img_pil)
                image_feature = get_image_features(img_tensor, model, device)
                class_idx = image_feature.argmax()  # 가장 높은 값의 인덱스를 클래스라 판단
                # 클래스를 인덱스로 저장하는 대신 class_names에서 클래스 텍스트를 가져와서 저장
                image_classes.append(class_names[class_idx.item()])  # 클래스 텍스트를 저장

    doc_type = 0
    if has_text and not has_image:
        doc_type = 1  # 텍스트만
    elif has_image and not has_text and not image_has_text:
        doc_type = 2  # 이미지만
    elif has_text and has_image and not image_has_text:
        doc_type = 3  # 텍스트+이미지
    elif has_image and not has_text and image_has_text:
        doc_type = 4  # 이미지 내 텍스트만
    elif has_text and has_image and image_has_text:
        doc_type = 5  # 텍스트+이미지+이미지 내 텍스트

    if return_pages:
        return doc_type, full_text.strip(), image_classes, page_texts, page_images
    return doc_type, full_text.strip(), image_classes
