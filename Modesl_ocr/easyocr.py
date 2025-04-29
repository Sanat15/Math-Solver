# math_solver_easyocr.py
import cv2
import easyocr
from sympy import Eq, symbols, solve

image_path = 'input.jpg'
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image not found. Check path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(thresh)
extracted = ' '.join([res[1] for res in results])
print("Extracted Text:", extracted)

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
