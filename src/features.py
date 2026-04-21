import pandas as pd
import re


def map_ebay_condition(raw: str) -> int:
    """Map eBay condition string to 1–5 numeric scale."""
    if not isinstance(raw, str):
        return 3
    t = raw.lower().strip()
    if any(k in t for k in ["brand new", "new with tags", "nwt", "deadstock"]):
        return 5
    if any(k in t for k in ["new (other)", "new without tags"]):
        return 4
    if "pre-owned" in t or "used" in t:
        return 3
    if any(k in t for k in ["for parts", "not working", "worn", "flaw"]):
        return 1
    return 3


def extract_condition(title: str) -> int:
    """Fallback: parse condition from listing title when no eBay field is available."""
    if not isinstance(title, str):
        return 3
    t = title.lower()
    if any(k in t for k in ["nwt", "new with tags", "new old stock", "nos", "deadstock", "unworn"]):
        return 5
    if any(k in t for k in ["mint", "near mint", " nm ", "excellent", "pristine"]):
        return 4
    if any(k in t for k in ["very good", "vg+", " vg ", "great condition", "nice condition"]):
        return 3
    if any(k in t for k in ["good condition", "light fade", "light wear", "gently worn", "some wear"]):
        return 2
    if any(k in t for k in [" faded", "heavy wear", "worn", "distressed", "thrashed"]):
        return 1
    return 3


def extract_features(title):
    title_lower = title.lower()
    
    features = {}
    
    # Era
    features["is_80s"] = 1 if "80s" in title_lower else 0
    features["is_90s"] = 1 if "90s" in title_lower else 0
    features["is_y2k"] = 1 if "y2k" in title_lower or "2000s" in title_lower else 0
    
    # Premium signals
    features["has_3d_emblem"] = 1 if re.search(r'3[-\s]?d[\s-]?emblem', title_lower) else 0
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

df = pd.read_csv("data/processed/harley_clean.csv")

feature_rows = []
for title in df["title"]:
    feature_rows.append(extract_features(title))

features_df = pd.DataFrame(feature_rows)

# Use scraped eBay condition field when available, fall back to title parsing
if "condition" in df.columns:
    features_df["condition"] = df["condition"].apply(map_ebay_condition)
else:
    features_df["condition"] = df["title"].apply(extract_condition)

# Use scraped size column directly — more accurate than title parsing
SIZE_ORDER = ["S", "M", "L", "XL", "2XL", "3XL", "4XL"]
if "size" in df.columns:
    features_df["size_s"]  = (df["size"] == "S").astype(int)
    features_df["size_m"]  = (df["size"] == "M").astype(int)
    features_df["size_l"]  = (df["size"] == "L").astype(int)
    features_df["size_xl"] = df["size"].isin(["XL", "2XL", "3XL", "4XL"]).astype(int)

features_df["sold_price"] = df["sold_price"]

print(features_df.head())
print(f"\nCondition distribution:\n{features_df['condition'].value_counts().sort_index()}")
features_df.to_csv("data/raw/harley_features.csv", index=False)
print("\nSaved to harley_features.csv")