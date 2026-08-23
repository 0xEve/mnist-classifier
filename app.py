import io

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from model import ClassModel

app = Flask(__name__)

device = torch.device("cpu")
model = ClassModel()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()


def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "مفيش صورة اتبعتت"}), 400

    file = request.files["image"]
    try:
        tensor = preprocess_image(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "الصورة مش قابلة للقراءة"}), 400

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted = int(torch.argmax(probs).item())

    return jsonify(
        {
            "predicted": predicted,
            "probabilities": [round(float(p), 4) for p in probs],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
