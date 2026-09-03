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
                background:
                    radial-gradient(circle at 15% 15%, rgba(99,102,241,0.18), transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(168,85,247,0.18), transparent 40%),
                    linear-gradient(135deg, #0b1023 0%, #171335 100%);
                min-height: 100vh; display: flex; flex-direction: column; justify-content: center;
                align-items: center; padding: 24px; color: #f8fafc;
            }
            .container {
                background: rgba(255, 255, 255, 0.045); backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.09); border-radius: 24px;
                padding: 44px 40px; width: 100%; max-width: 460px; text-align: center;
                box-shadow: 0 25px 60px rgba(0,0,0,0.45);
            }
            .icon-badge {
                width: 56px; height: 56px; margin: 0 auto 18px; border-radius: 16px;
                background: linear-gradient(135deg, #6366f1, #a855f7);
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 10px 25px rgba(99,102,241,0.35);
            }
            .icon-badge svg { width: 28px; height: 28px; fill: #fff; }
            h1 {
                font-size: 30px; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.5px;
                background: linear-gradient(to right, #a5b4fc, #e9d5ff);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }
            p.subtitle { font-size: 14px; color: #94a3b8; margin-bottom: 24px; }

            .notice {
                display: flex; align-items: flex-start; gap: 10px; text-align: left;
                background: rgba(250, 204, 21, 0.08); border: 1px solid rgba(250, 204, 21, 0.3);
                border-radius: 12px; padding: 14px 16px; margin-bottom: 26px;
            }
            .notice svg { flex-shrink: 0; width: 18px; height: 18px; fill: #facc15; margin-top: 2px; }
            .notice p { font-size: 12.5px; color: #fde68a; line-height: 1.55; margin: 0; }
            .notice strong { color: #fef9c3; }

            .drop-zone {
                display: flex; flex-direction: column; align-items: center; justify-content: center;
                gap: 12px; min-height: 160px;
                border: 2px dashed rgba(129, 140, 248, 0.5); border-radius: 16px; padding: 28px 20px;
                cursor: pointer; transition: all 0.25s ease; background: rgba(99, 102, 241, 0.04);
            }
            .drop-zone:hover { background: rgba(99, 102, 241, 0.12); border-color: #818cf8; transform: translateY(-1px); }
            .upload-icon-circle {
                width: 52px; height: 52px; border-radius: 50%;
                background: rgba(99, 102, 241, 0.15);
                display: flex; align-items: center; justify-content: center;
            }
            .upload-icon-circle svg { width: 24px; height: 24px; fill: #a5b4fc; }
            .drop-zone p { font-size: 13.5px; color: #cbd5e1; line-height: 1.5; }
            .drop-zone p.hint { font-size: 11.5px; color: #64748b; margin-top: -4px; }
            input[type="file"] { display: none; }
            .btn {
                width: 100%; margin-top: 22px; padding: 15px; border: none; border-radius: 12px;
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); color: white;
                font-weight: 600; font-size: 15.5px; cursor: pointer; transition: 0.25s;
                box-shadow: 0 8px 20px rgba(99,102,241,0.3);
            }
            .btn:hover { opacity: 0.92; transform: translateY(-2px); box-shadow: 0 12px 28px rgba(99,102,241,0.4); }
            #preview-container { display: none; flex-direction: column; align-items: center; gap: 10px; }
            #preview-img { max-width: 100px; max-height: 100px; border-radius: 10px; border: 2px solid #818cf8; background: #fff; padding: 6px; }
            #preview-container p { font-size: 12.5px; color: #94a3b8; }

            .footer { margin-top: 26px; font-size: 12px; color: #64748b; text-align: center; }
            .footer span { color: #a5b4fc; font-weight: 600; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon-badge">
                <svg viewBox="0 0 24 24"><path d="M9 3v2H4v14h16V5h-5V3H9zm3 4a5 5 0 110 10 5 5 0 010-10zm0 2a3 3 0 100 6 3 3 0 000-6z"/></svg>
            </div>
            <h1>DigitVision AI</h1>
            <p class="subtitle">Handwritten Digit Classification System</p>

            <div class="notice">
                <svg viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>
                <p><strong>Upload tip:</strong> Please upload an image of a digit (0-9) drawn with a <strong>dark pen/pencil on a plain white background</strong> for the most accurate prediction.</p>
            </div>

            <form action="/predict" method="post" enctype="multipart/form-data">
                <label for="file-input" class="drop-zone">
                    <div id="upload-prompt" style="display:flex;flex-direction:column;align-items:center;gap:12px;">
                        <div class="upload-icon-circle">
                            <svg viewBox="0 0 24 24"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
                        </div>
                        <p>Click to upload a digit image</p>
                        <p class="hint">White background &middot; dark digit &middot; PNG or JPG</p>
                    </div>
                    <div id="preview-container">
                        <img id="preview-img" src="" alt="Image Preview">
                        <p>Click to change image</p>
                    </div>
                </label>
                <input id="file-input" type="file" name="file" accept="image/*" required onchange="showPreview(event)">
                <button type="submit" class="btn">Classify Digit</button>
            </form>
        </div>

        <div class="footer">Created by <span>Jahid Hasan</span></div>

        <script>
            function showPreview(event) {
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('preview-img').src = e.target.result;
                        document.getElementById('preview-container').style.display = 'flex';
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
    # NOTE: Assumes a WHITE background image with a dark digit (standard photo/scan),
    # which is inverted here to match the MNIST-style white-digit-on-black-background format.
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
                background:
                    radial-gradient(circle at 15% 15%, rgba(99,102,241,0.18), transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(168,85,247,0.18), transparent 40%),
                    linear-gradient(135deg, #0b1023 0%, #171335 100%);
                min-height: 100vh; display: flex; flex-direction: column; justify-content: center;
                align-items: center; padding: 24px; color: #f8fafc;
            }}
            .container {{
                background: rgba(255, 255, 255, 0.045); backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.09); border-radius: 24px;
                padding: 44px 40px; width: 100%; max-width: 480px; text-align: center;
                box-shadow: 0 25px 60px rgba(0,0,0,0.45);
            }}
            .badge-label {{
                display: inline-block; font-size: 12px; letter-spacing: 0.5px; text-transform: uppercase;
                color: #a5b4fc; background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
                padding: 5px 14px; border-radius: 999px; margin-bottom: 18px;
            }}
            .digit-badge {{
                font-size: 84px; font-weight: 700; line-height: 1; color: #4ade80; margin: 4px 0 18px;
                text-shadow: 0 0 30px rgba(74,222,128,0.45);
            }}
            .images-grid {{ display: flex; justify-content: center; align-items: stretch; margin: 26px 0; gap: 16px; }}
            .img-card {{
                background: rgba(0,0,0,0.25); padding: 14px; border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.07); width: 130px;
            }}
            .img-card img {{ width: 90px; height: 90px; object-fit: contain; border-radius: 8px; margin-bottom: 8px; background: #fff; }}
            .img-card p {{ font-size: 11.5px; color: #94a3b8; }}
            .progress-bar {{ background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden; margin: 6px 0 8px 0; }}
            .progress-fill {{ background: linear-gradient(to right, #6366f1, #4ade80); height: 100%; width: {confidence}%; transition: width 0.4s ease; }}
            .confidence-text {{ font-size: 14px; color: #cbd5e1; margin-bottom: 4px; }}
            .btn {{
                display: inline-block; width: 100%; margin-top: 24px; padding: 15px; border-radius: 12px;
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); color: white;
                font-weight: 600; font-size: 15.5px; text-decoration: none; box-sizing: border-box;
                box-shadow: 0 8px 20px rgba(99,102,241,0.3); transition: 0.25s;
            }}
            .btn:hover {{ opacity: 0.92; transform: translateY(-2px); box-shadow: 0 12px 28px rgba(99,102,241,0.4); }}
            .footer {{ margin-top: 26px; font-size: 12px; color: #64748b; text-align: center; }}
            .footer span {{ color: #a5b4fc; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge-label">Prediction Result</span>
            <div class="digit-badge">{predicted_digit}</div>

            <div class="confidence-text">Confidence: <strong>{confidence}%</strong></div>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>

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

        <div class="footer">Created by <span>Jahid Hasan</span></div>
    </body>
    </html>
    """
