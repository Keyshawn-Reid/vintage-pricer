import os
import base64
import json
import io
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.brands import BRANDS

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_DIMENSION = 1600
JPEG_QUALITY = 85

def _compress_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

def extract_features_from_image(image_path, brand="harley"):
    image_data = base64.b64encode(_compress_image(image_path)).decode("utf-8")
    prompt = BRANDS[brand]["vision_prompt"]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }
        ],
        max_tokens=300
    )

    result = response.choices[0].message.content
    result = result.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(result)


if __name__ == "__main__":
    from src.model import predict_price_from_features
    features = extract_features_from_image("test.jpg", brand="harley")
    print("Extracted features:", features)
    low, high = predict_price_from_features(features)
    print(f"Estimated price range: ${low} - ${high}")
