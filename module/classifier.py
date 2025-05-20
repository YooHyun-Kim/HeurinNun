import argparse
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import os
import torch

# ResNet/DenseNet 모델 및 함수
from image_classifier.resnet import build_resnet, preprocess_image as resnet_preprocess, get_image_features as resnet_get_features
from image_classifier.densenet import build_densenet, preprocess_image as densenet_preprocess, get_image_features as densenet_get_features

# Tip-Adapter 예측 함수 가져오기
from module.tip_adapter.tip_adapter import predict_image_tip_adapter

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 분류할 클래스 이름 리스트
class_names = [
    "흐름도", "건축도면", "디바이스", "장비도면", "회로도면", "로고", "그래프"
]

# 전역 변수 초기화
model = None
preprocess_image = None
get_image_features = None


def initialize_model(model_type: str):
    global model, preprocess_image, get_image_features
    if model_type == 'resnet':
        model = build_resnet(feature_only=False).to(device).eval()
        preprocess_image = resnet_preprocess
        get_image_features = lambda img_tensor: resnet_get_features(img_tensor, model, device)
    elif model_type == 'densenet':
        model = build_densenet(feature_only=False).to(device).eval()
        preprocess_image = densenet_preprocess
        get_image_features = lambda img_tensor: densenet_get_features(img_tensor, model, device)
    elif model_type == 'tip_adapter':
        # Tip-Adapter는 predict_image_tip_adapter 함수 내부에서 CLIP과 메모리뱅크를 처리합니다.
        model = None
        preprocess_image = None
        get_image_features = None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def classify_pdf_document(pdf_path: str, return_pages: bool=False):
    doc = fitz.open(pdf_path)
    has_text = False
    has_image = False
    image_has_text = False
    full_text = ""

    page_texts = []
    page_images = []
    image_classes = []

    for page_num, page in enumerate(doc, start=1):
        # 텍스트 추출
        text = page.get_text().strip()
        if text:
            has_text = True
            full_text += text + "\n"
        page_texts.append(text)

        # 이미지 추출
        images = page.get_images(full=True)
        page_img_paths = []
        page_img_preds = []
        if images:
            has_image = True
            for img_idx, img_meta in enumerate(images, start=1):
                xref = img_meta[0]
                base_image = doc.extract_image(xref)
                img = Image.open(io.BytesIO(base_image['image'])).convert('RGB')

                # 파일로 저장
                out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
                os.makedirs(out_dir, exist_ok=True)
                img_path = os.path.join(out_dir, f"page_{page_num}_img_{img_idx}.png")
                img.save(img_path)
                page_img_paths.append(img_path)

                # OCR로 텍스트 유무
                if pytesseract.image_to_string(img).strip():
                    image_has_text = True

                # 이미지 분류
                try:
                    if model is not None:
                        # ResNet 또는 DenseNet 경로
                        tensor = preprocess_image(img)
                        feat = get_image_features(tensor)
                        idx = feat.argmax().item()
                    else:
                        # Tip-Adapter 경로
                        labels = predict_image_tip_adapter(img)
                        idx = class_names.index(labels[0])
                    page_img_preds.append(class_names[idx])
                except Exception as e:
                    print(f"⚠️ 분류 실패 page {page_num} img {img_idx}: {e}")
                    page_img_preds.append("UNKNOWN")

        page_images.append(page_img_paths)
        image_classes.append(page_img_preds)

    # 문서 유형 판단 로직 (기존 유지)
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
    return doc_type, full_text.strip(), image_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDF 분류기 (ResNet, DenseNet, Tip-Adapter)')
    parser.add_argument('--model', choices=['resnet','densenet','tip_adapter'], default='resnet',
                        help='사용할 이미지 분류 백본')
    parser.add_argument('--pdf', required=True, help='PDF 파일 경로')
    parser.add_argument('--return_pages', action='store_true',
                        help='페이지별 세부정보 반환')
    args = parser.parse_args()

    initialize_model(args.model)
    result = classify_pdf_document(args.pdf, return_pages=args.return_pages)
    print(result)