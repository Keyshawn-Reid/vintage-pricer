import pandas as pd
import re

def extract_features(title):
    title_lower = title.lower()
    
    features = {}
    
    # Era
    features["is_80s"] = 1 if "80s" in title_lower else 0
    features["is_90s"] = 1 if "90s" in title_lower else 0
    features["is_y2k"] = 1 if "y2k" in title_lower or "2000s" in title_lower else 0
    
    # Premium signals
    features["has_3d_emblem"] = 1 if "3d emblem" in title_lower else 0
    features["has_single_stitch"] = 1 if "single stitch" in title_lower else 0
    features["is_vtg"] = 1 if "vtg" in title_lower or "vintage" in title_lower else 0
    
    # Size
    features["size_s"] = 1 if " s " in title_lower or "small" in title_lower else 0
    features["size_m"] = 1 if " m " in title_lower or "medium" in title_lower else 0
    features["size_l"] = 1 if " l " in title_lower or "large" in title_lower else 0
    features["size_xl"] = 1 if "xl" in title_lower or "extra large" in title_lower else 0

    # High value signals
    features["has_location"] = 1 if any(city in title_lower for city in [
        "florida", "california", "texas", "arizona", "alaska", "daytona", 
        "sturgis", "milwaukee", "pomona"
    ]) else 0

    features["is_event_tee"] = 1 if any(word in title_lower for word in [
        "rally", "run", "fest", "classic", "event"
    ]) else 0

    features["has_year"] = 1 if re.search(r'\b(19[6-9]\d|200\d)\b', title_lower) else 0

    return features

df = pd.read_csv("data/raw/harley_raw.csv")

feature_rows = []

for title in df["title"]:
    f = extract_features(title)
    feature_rows.append(f)

features_df = pd.DataFrame(feature_rows)
features_df["sold_price"] = df["sold_price"]

print(features_df.head())
features_df.to_csv("data/raw/harley_features.csv", index=False)
print("Saved to harley_features.csv")