# math_solver_trocr.py
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sympy import Eq, symbols, solve

image_path = 'input.jpg'
image = Image.open(image_path).convert('RGB')

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

pixel_values = processor(images=image, return_tensors="pt").pixel_values
with torch.no_grad():
    generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Predicted Text:", text)

# Solve
x = symbols('x')
try:
    cleaned = text.replace("x", "*x") if 'x' in text and '*' not in text else text
    cleaned = cleaned.replace(" ", "").strip()
    lhs, rhs = cleaned.split('=')
    equation = Eq(eval(lhs), eval(rhs))
    solution = solve(equation, x)
    print("Solution:", solution)
except Exception as e:
    print("Error while parsing:", e)
