import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

import sys
sys.path.append("/Users/keyshawnreid/vintage-pricer")
from src.model import predict_price_from_features

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_features_from_image(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = """Look at this vintage Harley Davidson tee. 
    Extract the following and return as JSON only, no other text:
    {
        "era": "80s or 90s or y2k or unknown",
        "size": "S or M or L or XL or unknown",
        "has_3d_emblem": true or false,
        "has_single_stitch": true or false,
        "has_location_name": true or false,
        "is_event_tee": true or false
    }"""

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
    import sys
    sys.path.append("/Users/keyshawnreid/vintage-pricer")
    from src.model import predict_price_from_features
    
    features = extract_features_from_image("test.jpg")
    print("Extracted features:", features)
    
    low, high = predict_price_from_features(features)
    print(f"Estimated price range: ${low} - ${high}")