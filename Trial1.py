# Handwritten Math Solver Pipeline (Improved Version)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, simplify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------- Step 1: Train on Custom Dataset from IMG Folder --------
def build_symbol_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(15, activation='softmax')  # 10 digits + 5 operators
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_custom_dataset(folder='IMG'):
    X, y = [], []
    label_map = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'plus': 10, 'minus': 11, 'mul': 12, 'div': 13, 'equal': 14
    }
    for label_name in os.listdir(folder):
        label_folder = os.path.join(folder, label_name)
        if not os.path.isdir(label_folder):
            continue
        label = label_map.get(label_name)
        if label is None:
            continue
        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)
            img = load_img(img_path, target_size=(28, 28), color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label)
    return np.array(X), to_categorical(y, num_classes=15)

def train_symbol_model():
    x, y = load_custom_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = build_symbol_model()
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save("symbol_model.h5")
    return model

# -------- Step 2: Preprocess Image and Extract Symbols --------
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbols = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, w, h))

    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    for (x, y, w, h) in bounding_boxes:
        symbol = thresh[y:y+h, x:x+w]
        symbol = cv2.resize(symbol, (28, 28))
        symbol = symbol / 255.0
        symbols.append(symbol)

    return np.array(symbols).reshape(-1, 28, 28, 1)

# -------- Step 3: Recognize Symbols --------
def recognize_symbols(model, symbol_images):
    predictions = model.predict(symbol_images)
    labels = np.argmax(predictions, axis=1)
    return labels

# -------- Step 4: Construct Equation --------
def construct_equation(label_list):
    label_map = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '+', 11: '-', 12: '*', 13: '/', 14: '='
    }
    expr = ''.join(label_map.get(label, '?') for label in label_list)
    return expr

# -------- Step 5: Solve Using SymPy --------
def solve_equation(expr_str):
    x = symbols('x')
    if '=' in expr_str:
        left, right = expr_str.split('=')
        eq = Eq(simplify(left), simplify(right))
        return solve(eq, x)
    return "Invalid equation format"

# -------- Example Usage --------
if __name__ == '__main__':
    # Step 1: Load or Train model
    if os.path.exists("symbol_model.h5"):
        model = load_model("symbol_model.h5")
    else:
        model = train_symbol_model()

    # Step 2: Process image
    image_path = 'sample_math_img.png'  # Replace with your image path
    symbols_array = preprocess_image(image_path)

    # Step 3: Recognize digits/operators
    labels = recognize_symbols(model, symbols_array)
    print("Recognized labels:", labels)

    # Step 4: Construct equation
    expr = construct_equation(labels)
    print("Constructed Equation:", expr)

    # Step 5: Solve equation
    try:
        result = solve_equation(expr)
        print("Solution:", result)
    except Exception as e:
        print("Error solving equation:", e)
