from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# 🔥 IMPORTANT: EfficientNet preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# ===============================
# Load Model (ONCE at startup)
# ===============================
MODEL_PATH = "final_efficientnetb3_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# Load Class Indices
# ===============================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class label
idx_to_class = {v: k for k, v in class_indices.items()}

# ===============================
# Image Settings
# ===============================
IMG_SIZE = 300  # EfficientNetB3 input size

# ===============================
# Preprocess Image (MATCH TRAINING)
# ===============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    # ✅ SAME preprocessing used during training
    image = preprocess_input(image)

    return image

# ===============================
# Health Check
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Model API is running 🚀",
        "model": "EfficientNetB3",
        "total_classes": len(idx_to_class)
    })

# ===============================
# Prediction Endpoint
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate input
        if "image" not in request.files:
            return jsonify({"error": "Image file not provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Load & preprocess image
        image = Image.open(file)
        processed_image = preprocess_image(image)

        # Predict
        preds = model.predict(processed_image, verbose=0)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Response
        return jsonify({
            "predicted_class": idx_to_class[class_id],
            "confidence_percent": round(confidence * 100, 2)
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# ===============================
# Run App (Render compatible)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
