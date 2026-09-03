import io
import fastapi
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps

app = FastAPI()

# Load model
model = tf.keras.models.load_model("model.h5")

# Basic HTML frontend template
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Handwritten Digit Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f4f9; }
        .card { background: white; max-width: 400px; margin: auto; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        input { margin: 15px 0; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Handwritten Digit Classifier</h2>
        <form action="/predict" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*" required>
            <br>
            <button type="submit">Predict Digit</button>
        </form>
        {% if prediction is not none %}
            <h3>Predicted Digit: {{ prediction }}</h3>
            <p>Confidence: {{ confidence }}%</p>
        {% endif %}
    </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return html_content.replace("{% if prediction is not none %}", "").replace(
        "{% endif %}", ""
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess image for model (28x28 grayscale, inverted background)
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    res = model.predict(img_array)[0]
    predicted_digit = int(np.argmax(res))
    confidence = round(float(np.max(res)) * 100, 2)

    output_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Handwritten Digit Predictor</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f4f9; }}
            .card {{ background: white; max-width: 400px; margin: auto; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            button {{ padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Handwritten Digit Classifier</h2>
            <h1 style="color: #28a745; font-size: 48px;">{{ predicted_digit }}</h1>
            <p>Confidence: {{ confidence }}%</p>
            <br>
            <a href="/"><button>Predict Another Image</button></a>
        </div>
    </body>
    </html>
    """
    return (
        output_html.replace("{{ predicted_digit }}", str(predicted_digit))
        .replace("{{ confidence }}", str(confidence))
    )