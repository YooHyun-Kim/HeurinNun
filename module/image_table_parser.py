from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import cv2
import numpy as np

def parse_table_from_pdf(pdf_path, page_num=0, lang="eng+kor", y_threshold=10):
    """
    PDF 파일에서 표를 인식하여 DataFrame으로 반환
    """
    # 1. PDF 페이지를 이미지로 변환
    pages = convert_from_path(pdf_path)
    img_pil = pages[page_num]  # 지정한 페이지 하나만

    # 2. PIL → OpenCV 이미지로 변환
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 3. OCR 데이터프레임
    ocr_df = pytesseract.image_to_data(img_cv, lang=lang, output_type=pytesseract.Output.DATAFRAME)
    ocr_df = ocr_df.dropna(subset=["text"])
    ocr_df = ocr_df[ocr_df.conf != -1]

    rows = []
    current_line = []
    last_y = None

    for _, row in ocr_df.iterrows():
        y = row["top"]
        text = row["text"]
        if last_y is None or abs(y - last_y) < y_threshold:
            current_line.append(text)
        else:
            rows.append(current_line)
            current_line = [text]
        last_y = y

    if current_line:
        rows.append(current_line)

    return pd.DataFrame(rows)


