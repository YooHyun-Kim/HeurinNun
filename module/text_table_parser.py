import pdfplumber
import pandas as pd

def extract_tables_from_pdf(pdf_path):
    """
    텍스트 기반 PDF에서 표 추출
    Returns: List of DataFrames
    """
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            extracted = page.extract_table()
            if extracted:
                df = pd.DataFrame(extracted[1:], columns=extracted[0])
                tables.append((i, df))  # 페이지 번호와 함께 반환

    return tables
