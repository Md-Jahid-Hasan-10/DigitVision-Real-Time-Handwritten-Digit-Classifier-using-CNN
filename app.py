import base64
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps

app = FastAPI()

# Load Trained Model
model = tf.keras.models.load_model("model.h5")


def get_base64_image(pil_img):
    """Utility to convert PIL Image to base64 data URI for direct inline HTML display."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DigitVision AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            body { 
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
                min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; color: #f8fafc; 
            }
            .container { 
                background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(16px); 
                border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; 
                padding: 40px; width: 100%; max-width: 480px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.4); 
            }
            h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            p.subtitle { font-size: 14px; color: #94a3b8; margin-bottom: 30px; }
            .drop-zone { 
                border: 2px dashed #6366f1; border-radius: 12px; padding: 30px 20px; 
                cursor: pointer; transition: 0.3s; background: rgba(99, 102, 241, 0.05); 
            }
            .drop-zone:hover { background: rgba(99, 102, 241, 0.15); border-color: #818cf8; }
            input[type="file"] { display: none; }
            .btn { 
                width: 100%; margin-top: 20px; padding: 14px; border: none; border-radius: 10px; 
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); color: white; 
                font-weight: 600; font-size: 16px; cursor: pointer; transition: 0.3s; 
            }
            .btn:hover { opacity: 0.9; transform: translateY(-1px); }
            #preview-container { display: none; margin-top: 15px; }
            #preview-img { max-width: 120px; border-radius: 8px; border: 2px solid #818cf8; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DigitVision AI</h1>
            <p class="subtitle">Handwritten Digit Classification System</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <label for="file-input" class="drop-zone">
                    <div id="upload-prompt">
                        <svg style="width:40px;height:40px;fill:#818cf8;" viewBox="0 0 24 24"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
                        <p style="margin-top:8px;font-size:14px;">Click to upload an image of a digit (0-9)</p>
                    </div>
                    <div id="preview-container">
                        <img id="preview-img" src="" alt="Image Preview">
                    </div>
                </label>
                <input id="file-input" type="file" name="file" accept="image/*" required onchange="showPreview(event)">
                <button type="submit" class="btn">Classify Digit</button>
            </form>
        </div>
        <script>
            function showPreview(event) {
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('preview-img').src = e.target.result;
                        document.getElementById('preview-container').style.display = 'block';
                        document.getElementById('upload-prompt').style.display = 'none';
                    }
                    reader.readAsDataURL(file);
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    raw_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Generate base64 string for displaying original image
    raw_image_b64 = get_base64_image(raw_image)

    # Preprocessing Pipeline (Grayscale -> Invert -> Resize -> Normalize)
    processed_img = ImageOps.grayscale(raw_image)
    processed_img = ImageOps.invert(processed_img)
    processed_img = processed_img.resize((28, 28))

    # Convert processed 28x28 image to base64 for display
    processed_image_b64 = get_base64_image(processed_img)

    # Convert to array for Keras Model
    img_array = np.array(processed_img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Model Inference
    predictions = model.predict(img_array)[0]
    predicted_digit = int(np.argmax(predictions))
    confidence = round(float(np.max(predictions)) * 100, 2)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Result</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
            body {{ 
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
                min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; color: #f8fafc; 
            }}
            .container {{ 
                background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(16px); 
                border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; 
                padding: 40px; width: 100%; max-width: 500px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.4); 
            }}
            .digit-badge {{ font-size: 72px; font-weight: 700; color: #22c55e; margin: 10px 0; text-shadow: 0 0 20px rgba(34,197,94,0.4); }}
            .images-grid {{ display: flex; justify-content: space-around; align-items: center; margin: 25px 0; gap: 15px; }}
            .img-card {{ background: rgba(0,0,0,0.2); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); width: 45%; }}
            .img-card img {{ width: 80px; height: 80px; object-fit: contain; border-radius: 6px; margin-bottom: 6px; }}
            .img-card p {{ font-size: 12px; color: #94a3b8; }}
            .progress-bar {{ background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden; margin: 15px 0 5px 0; }}
            .progress-fill {{ background: linear-gradient(to right, #6366f1, #22c55e); height: 100%; width: {confidence}%; }}
            .btn {{ 
                display: inline-block; width: 100%; margin-top: 20px; padding: 14px; border-radius: 10px; 
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); color: white; 
                font-weight: 600; font-size: 16px; text-decoration: none; box-sizing: border-box; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style="font-size:18px; color:#94a3b8; font-weight:400;">Predicted Digit</h2>
            <div class="digit-badge">{predicted_digit}</div>

            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <p style="font-size: 14px; color: #cbd5e1;">Confidence: <strong>{confidence}%</strong></p>

            <div class="images-grid">
                <div class="img-card">
                    <img src="{raw_image_b64}" alt="Uploaded Image">
                    <p>Uploaded Image</p>
                </div>
                <div class="img-card">
                    <img src="{processed_image_b64}" alt="Model Input">
                    <p>Model Input (28x28)</p>
                </div>
            </div>

            <a href="/" class="btn">Classify Another Image</a>
        </div>
    </body>
    </html>
    """
