import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("downstream_model_weights.h5")

class_labels = {
    0: '2S1', 1: 'BMP2', 2: 'BRDM2', 3: 'BTR60',
    4: 'BTR70', 5: 'D7', 6: 'SLICY', 7: 'T62',
    8: 'T72', 9: 'ZIL132', 10: 'ZSU_23_4'
}

def preprocess(img):
    img = img.convert("RGB")                      # RGB
    img = img.resize((224, 224))                 # Resize like notebook
    img = np.array(img).astype("float32")        # To float32
    img = img / 255.0                             # Normalize
    img = np.expand_dims(img, axis=0)            # Batch dimension
    return img

def predict_image(inp):
    img = preprocess(inp)
    pred = model.predict(img)
    cls = np.argmax(pred[0])
    return class_labels[cls]

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="SAR ATR Classifier"
)

interface.launch()