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
from src.model import predict_price_from_features

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

def extract_features_from_image(image_path):
    image_data = base64.b64encode(_compress_image(image_path)).decode("utf-8")

    prompt = """You are an expert vintage Harley Davidson t-shirt appraiser. 
    Analyze this image carefully and extract the following signals.
    Return ONLY valid JSON, no other text:
    {
        "era": "80s or 90s or y2k or unknown",
        "size": "S or M or L or XL or 2XL or unknown",
        "has_3d_emblem": true or false,
        "has_single_stitch": true or false,
        "has_location_name": true or false,
        "is_event_tee": true or false
    }

    Rules:
    - 3D emblem means the Harley shield logo appears raised/embossed on the graphic
    - Single stitch means the sleeve hems have single row stitching (vintage indicator)
    - Location name means a specific city or state appears on the shirt
    - Event tee means it references a rally, run, or specific event
    - For era: 80s graphics tend to be bolder/simpler, 90s are more detailed/airbrushed
    - Look at tags if visible for size and era clues"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )

    result = response.choices[0].message.content
    result = result.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(result)


if __name__ == "__main__":
    
    features = extract_features_from_image("test.jpg")
    print("Extracted features:", features)
    
    low, high = predict_price_from_features(features)
    print(f"Estimated price range: ${low} - ${high}")