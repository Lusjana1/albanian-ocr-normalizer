from paddleocr import PaddleOCR

# Initialize OCR model
ocr_model = PaddleOCR(lang='sq')

def extract_text(image_path: str) -> str:
    result = ocr_model.ocr(image_path, cls=True)
    text = ""
    for line in result:
        for item in line:
            text += item[1][0] + " "
    return text.strip()