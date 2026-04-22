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

_DETECT_PROMPT = (
    'Identify the vintage clothing brand. Return ONLY valid JSON: '
    '{"detected_brand": "harley or ed_hardy or hysteric or unknown"}\n'
    'harley = Harley Davidson (eagle/shield/HOG/motorcycle graphics)\n'
    'ed_hardy = Ed Hardy (Don Ed Hardy tattoo art, rhinestones, Christian Audigier)\n'
    'hysteric = Hysteric Glamour (Japanese streetwear, devil babe, pin-up, snake, Japanese tags)'
)

_SLOT_LABELS = {
    "front": "FRONT — main graphic side of the garment",
    "tag":   "TAG — collar or neck tag (use for era, brand line, Japan domestic, collab text, tag generation)",
    "back":  "BACK — back of the garment (check for back graphic)",
    "care":  "CARE LABEL — wash/care label (country of origin, fiber content, possible year codes)",
}


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


def extract_features_from_images(image_paths: dict, brand: str = "hysteric") -> dict:
    """Multi-image extraction. image_paths = {slot: path} where slot is front/tag/back/care."""
    prompt = BRANDS[brand]["vision_prompt"]

    role_lines = [
        f"Image {i + 1} is the {_SLOT_LABELS[slot]}."
        for i, slot in enumerate(image_paths)
        if slot in _SLOT_LABELS
    ]
    role_preamble = "\n".join(role_lines)

    content = [{"type": "text", "text": f"{role_preamble}\n\n{prompt}"}]
    for path in image_paths.values():
        image_data = base64.b64encode(_compress_image(path)).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=400,
    )

    result = response.choices[0].message.content.strip()
    result = result.replace("```json", "").replace("```", "").strip()
    return json.loads(result)


def detect_brand(image_paths: dict) -> str:
    """Single fast call using only the front image to identify brand."""
    path = image_paths.get("front") or next(iter(image_paths.values()), None)
    if not path:
        return "harley"
    image_data = base64.b64encode(_compress_image(path)).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _DETECT_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ]}],
        max_tokens=50,
    )
    result = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    detected = json.loads(result).get("detected_brand", "unknown")
    from src.brands import BRANDS
    return detected if detected in BRANDS else "harley"


if __name__ == "__main__":
    from src.model import predict_price_from_features
    features = extract_features_from_image("test.jpg", brand="harley")
    print("Extracted features:", features)
    low, high = predict_price_from_features(features)
    print(f"Estimated price range: ${low} - ${high}")
