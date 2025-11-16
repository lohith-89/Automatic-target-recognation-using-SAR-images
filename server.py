import base64
import numpy as np
import io
import gdown
import os
from PIL import Image
import tensorflow as tf
from flask import request, jsonify, Flask, render_template, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = "your_secret_key_here"

# -------------------------------------------------
# DOWNLOAD MODEL WEIGHTS FROM GOOGLE DRIVE IF MISSING
# -------------------------------------------------
MODEL_PATH = "downstream_model_weights.h5"
MODEL_URL = "https://drive.google.com/uc?id=1kxcFY-roQFFr3nw_uaEf2siW_XE3xWC4"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    print("Download complete!")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# -------------------------------------------------
# CLASS LABELS
# -------------------------------------------------
class_labels = {
    0: '2S1', 1: 'BMP2', 2: 'BRDM2', 3: 'BTR60',
    4: 'BTR70', 5: 'D7', 6: 'SLICY', 7: 'T62',
    8: 'T72', 9: 'ZIL132', 10: 'ZSU_23_4'
}

# -------------------------------------------------
# VEHICLE DETAILS
# -------------------------------------------------
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

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/gallery")
def gallery():
    return render_template("gallery.html", vehicle_details=vehicle_details)

@app.route("/history")
def history():
    history_list = session.get("prediction_history", [])
    return render_template("history.html", history=history_list)

# -------------------------------------------------
# PREDICTION API
# -------------------------------------------------
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

        # Encode uploaded image
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

# -------------------------------------------------
# RUN FLASK
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render dynamically sets PORT
    app.run(host="0.0.0.0", port=port)
