import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from flask import request, jsonify, Flask, render_template, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key_here'  # Change this to a secure key in production

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("downstream_model_weights.h5")

# -----------------------------
# Class Labels
# -----------------------------
class_labels = {
    0: '2S1', 1: 'BMP2', 2: 'BRDM2', 3: 'BTR60',
    4: 'BTR70', 5: 'D7', 6: 'SLICY', 7: 'T62',
    8: 'T72', 9: 'ZIL132', 10: 'ZSU_23_4'
}

# -----------------------------
# Vehicle Details Dictionary
# -----------------------------
vehicle_details = {
    '2S1': {
        'description': 'The 2S1 Gvozdika is a Soviet self-propelled artillery vehicle with a 122mm howitzer.',
        'image_url': '/static/images/2s1.jpeg'
    },
    'BMP2': {
        'description': 'The BMP-2 is a Soviet infantry fighting vehicle with a 30mm autocannon.',
        'image_url': '/static/images/bmp2.jpeg'
    },
    'BRDM2': {
        'description': 'BRDM-2 is an amphibious armored scout car designed for reconnaissance.',
        'image_url': '/static/images/bmrd2.jpeg'
    },
    'BTR60': {
        'description': 'The BTR-60 is an 8-wheeled armored personnel carrier from the Soviet era.',
        'image_url': '/static/images/btr60.jpeg'
    },
    'BTR70': {
        'description': 'An upgraded Soviet APC with improved armor and weapon system.',
        'image_url': '/static/images/btr70.jpeg'
    },
    'D7': {
        'description': 'The D7 is a bulldozer used for military engineering and earthmoving operations.',
        'image_url': '/static/images/d7.jpeg'
    },
    'SLICY': {
        'description': 'SLICY is a standard target type used in SAR datasets for ATR research.',
        'image_url': '/static/images/slicy.jpg'
    },
    'T62': {
        'description': 'The T-62 is a Soviet main battle tank known for its 115mm smoothbore gun.',
        'image_url': '/static/images/t62.jpeg'
    },
    'T72': {
        'description': 'The T-72 is a globally used main battle tank with a 125mm gun.',
        'image_url': '/static/images/t72.jpeg'
    },
    'ZIL132': {
        'description': 'The ZIL-132 is a heavy-duty Soviet military truck used for logistics.',
        'image_url': '/static/images/zil132.jpeg'
    },
    'ZSU_23_4': {
        'description': 'The ZSU-23-4 Shilka is a self-propelled anti-aircraft weapon system.',
        'image_url': '/static/images/zsu234.jpeg'
    }
}

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")

# -----------------------------
# Predict Route (Page)
# -----------------------------
@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# -----------------------------
# About Route
# -----------------------------
@app.route("/about")
def about():
    return render_template("about.html")

# -----------------------------
# Gallery Route
# -----------------------------
@app.route("/gallery")
def gallery():
    return render_template("gallery.html", vehicle_details=vehicle_details)

# -----------------------------
# History Route
# -----------------------------
@app.route("/history")
def history():
    history_list = session.get('prediction_history', [])
    return render_template("history.html", history=history_list)

# -----------------------------
# Prediction API Route
# -----------------------------
# -----------------------------
# Prediction API Route (FIXED)
# -----------------------------
@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)

        label = class_labels[predicted_index]
        details = vehicle_details[label]

        # Base64 encoding
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded_uploaded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "predicted_class": label,
            "description": details["description"],
            "vehicle_image_url": details["image_url"],
            "uploaded_image": encoded_uploaded
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)