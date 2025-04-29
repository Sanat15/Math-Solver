# math_solver_tesseract.py
import cv2
import pytesseract
from sympy import Eq, symbols, solve

# Load and preprocess image
image_path = 'input.jpg'
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image not found. Check path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
thresh = cv2.dilate(thresh, kernel, iterations=1)

# OCR
custom_config = r'--oem 3 --psm 7'
extracted = pytesseract.image_to_string(thresh, config=custom_config)
print("Extracted Text:", extracted.strip())

# Solve
x = symbols('x')
try:
    cleaned = extracted.replace("x", "*x") if 'x' in extracted and '*' not in extracted else extracted
    cleaned = cleaned.replace(" ", "").strip()
    lhs, rhs = cleaned.split('=')
    equation = Eq(eval(lhs), eval(rhs))
    solution = solve(equation, x)
    print("Solution:", solution)
except Exception as e:
    print("Error while parsing:", e)
