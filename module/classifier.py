import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from image_classifier.resnet import build_resnet
from image_classifier.densenet import build_densenet
from image_classifier.efficientnet import build_efficientnet


def classify_pdf_document(pdf_path):
    doc = fitz.open(pdf_path)
    
    has_text = False
    has_image = False
    image_has_text = False
    full_text = ""

    for page in doc:
        # 1. 텍스트 추출
        text = page.get_text()
        if text.strip():
            has_text = True
            full_text += text + "\n"

        # 2. 이미지 + OCR
        images = page.get_images(full=True)
        if images:
            has_image = True
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))

                # OCR 전처리 생략 가능
                if not image_has_text:
                    ocr_result = pytesseract.image_to_string(img_pil)
                    if ocr_result.strip():
                        image_has_text = True
                        break

    # 분류
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
    else:
        doc_type = 0

    return doc_type, full_text.strip()

model = build_resnet(num_classes=4)  # 바꾸고 싶으면 여기만 바꾸면 됨