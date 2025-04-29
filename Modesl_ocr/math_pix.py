# math_solver_mathpix.py
import base64
import requests
from sympy import Eq, symbols, solve

image_path = 'input.jpg'
with open(image_path, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

r = requests.post(
    url="https://api.mathpix.com/v3/text",
    headers={
        "app_id": "your_app_id",
        "app_key": "your_app_key",
        "Content-type": "application/json"
    },
    json={
        "src": f"data:image/jpg;base64,{img_base64}",
        "formats": ["text", "data"],
        "rm_spaces": True
    }
)

text = r.json().get("text", "")
print("MathPix Output:", text)

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
