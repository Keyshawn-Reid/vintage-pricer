import pandas as pd
import re


def map_condition(raw: str) -> int:
    if not isinstance(raw, str):
        return 3
    t = raw.lower().strip()
    if any(k in t for k in ["brand new", "new with tags", "nwt", "deadstock", "new"]):
        return 5
    if any(k in t for k in ["new (other)", "new without tags"]):
        return 4
    if any(k in t for k in ["gently used", "gently worn", "pre-owned"]):
        return 3
    if "used" in t:
        return 2
    if any(k in t for k in ["for parts", "flaw", "worn", "damaged"]):
        return 1
    return 3


def extract_features(title: str) -> dict:
    t = title.lower()

    features = {}

    # ── Era ───────────────────────────────────────────────────────────────────
    features["is_80s"]     = 1 if "80s" in t or re.search(r'\b198\d\b', t) else 0
    features["is_90s"]     = 1 if "90s" in t or re.search(r'\b199\d\b', t) else 0
    features["is_y2k"]     = 1 if any(k in t for k in ["y2k", "2000s", "00s"]) or re.search(r'\b200[0-9]\b', t) else 0
    features["is_vintage"]  = 1 if any(k in t for k in ["vintage", "vtg"]) else 0

    # ── Collab tier ───────────────────────────────────────────────────────────
    features["is_supreme_collab"] = 1 if "supreme" in t else 0

    BAND_ARTISTS = [
        "ramones", "the cramps", "cramps", "runaways", "cherry bomb",
        "kurt cobain", "courtney love", "hole ", "nirvana",
        "sex pistols", "iggy pop", "blondie",
    ]
    features["is_band_collab"] = 1 if any(k in t for k in BAND_ARTISTS) else 0

    OTHER_COLLABS = [
        "playboy", "andy warhol", "warhol", "richardson",
        "undercover", "madhappy", "vanson",
    ]
    features["is_collab"] = 1 if (
        features["is_supreme_collab"] or
        features["is_band_collab"] or
        any(k in t for k in OTHER_COLLABS)
    ) else 0

    # ── Origin ────────────────────────────────────────────────────────────────
    features["is_made_in_japan"] = 1 if any(k in t for k in ["made in japan", "mij", "japanese"]) else 0

    # ── Category ─────────────────────────────────────────────────────────────
    features["cat_tee"]      = 1 if any(k in t for k in ["tee", "t-shirt", "tank", "camisole"]) else 0
    features["cat_hoodie"]   = 1 if any(k in t for k in ["hoodie", "zip-up", "zip up", "sweatshirt", "sweater"]) else 0
    features["cat_jacket"]   = 1 if any(k in t for k in ["jacket", "bomber", "blouson"]) else 0
    features["cat_denim"]    = 1 if any(k in t for k in ["denim", "jeans", "pants", "flare"]) else 0
    features["cat_knitwear"] = 1 if any(k in t for k in ["knit", "cardigan", "crochet"]) else 0
    features["cat_shirt"]    = 1 if any(k in t for k in ["shirt", "polo", "button", "long sleeve"]) and not features["cat_tee"] else 0

    # ── Graphic signals ───────────────────────────────────────────────────────
    features["has_skull_graphic"] = 1 if any(k in t for k in ["skull", "skeleton", "bones"]) else 0
    features["has_girl_graphic"]  = 1 if any(k in t for k in ["girl", "vixen", "pin-up", "pinup", "pam"]) else 0
    features["has_logo_print"]    = 1 if any(k in t for k in ["logo", "multi logo", "all over"]) else 0

    return features


df = pd.read_csv("data/processed/hysteric_clean.csv")

feature_rows = []
for title in df["title"]:
    feature_rows.append(extract_features(title))

features_df = pd.DataFrame(feature_rows)

features_df["condition"] = df["condition"].apply(map_condition)

SIZE_MAP = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "2XL": 5, "3XL": 5}
if "size" in df.columns:
    features_df["size_xs"] = (df["size"] == "XS").astype(int)
    features_df["size_s"]  = (df["size"] == "S").astype(int)
    features_df["size_m"]  = (df["size"] == "M").astype(int)
    features_df["size_l"]  = (df["size"] == "L").astype(int)
    features_df["size_xl"] = df["size"].isin(["XL", "2XL", "3XL"]).astype(int)

features_df["sold_price"] = df["sold_price"]

print(features_df.head())
print(f"\nCondition distribution:\n{features_df['condition'].value_counts().sort_index()}")
print(f"\nCollab breakdown:")
print(f"  Supreme:    {features_df['is_supreme_collab'].sum()}")
print(f"  Band/Artist:{features_df['is_band_collab'].sum()}")
print(f"  Any collab: {features_df['is_collab'].sum()}")

features_df.to_csv("data/raw/hysteric_features.csv", index=False)
print("\nSaved → data/raw/hysteric_features.csv")
