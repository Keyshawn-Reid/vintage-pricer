from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.vision import extract_features_from_image, extract_features_from_images, detect_brand
from src.hysteric_rules import predict_price as hysteric_rules_predict
from src.brands import BRANDS
from src.feedback import compute_image_ref, save_feedback

app = Flask(__name__)

# ── Train one model per brand on startup ────────────────────────────────────
_models = {}

for _brand_id, _cfg in BRANDS.items():
    _csv = _cfg["csv"]
    if not os.path.exists(_csv):
        print(f"[{_brand_id}] no CSV found — model skipped")
        continue
    _df = pd.read_csv(_csv)
    _X = _df.drop("sold_price", axis=1)
    _y = np.log1p(_df["sold_price"])
    _m = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    _m.fit(_X, _y)
    _models[_brand_id] = {"model": _m, "columns": list(_X.columns)}
    print(f"[{_brand_id}] model trained on {len(_df)} rows")


# ── Feature extraction: form → input DataFrame ──────────────────────────────
def form_to_input_df(brand, data):
    print(f"[RPM form_data] brand={brand} raw={dict(data)}", flush=True)
    era = data.get("era", "unknown")
    size = data.get("size", "unknown")

    if brand == "harley":
        row = {
            "is_80s": int(era == "80s"),
            "is_90s": int(era == "90s"),
            "is_y2k": int(era == "y2k"),
            "has_3d_emblem": int(data.get("emblem", "0") == "1"),
            "has_single_stitch": int(data.get("single_stitch", "0") == "1"),
            "is_vtg": 1,
            "size_s": int(size == "S"),
            "size_m": int(size == "M"),
            "size_l": int(size == "L"),
            "size_xl": int(size == "XL"),
            "has_location": int(data.get("location", "0") == "1"),
            "is_event_tee": int(data.get("event", "0") == "1"),
            "has_year": 0,
            "condition": int(data.get("condition", "3")),
        }

    elif brand == "ed_hardy":
        row = {
            "is_y2k": int(era == "y2k"),
            "is_10s": int(era == "10s"),
            "size_s": int(size == "S"),
            "size_m": int(size == "M"),
            "size_l": int(size == "L"),
            "size_xl": int(size == "XL"),
            "size_2xl": int(size == "2XL"),
            "has_rhinestones": int(data.get("rhinestones", "0") == "1"),
            "is_collab": int(data.get("collab", "0") == "1"),
            "is_designer_series": int(data.get("designer_series", "0") == "1"),
            "is_vtg": 1,
        }

    elif brand == "hysteric":
        graphic = data.get("graphic_type", "unknown")
        collab_tier = data.get("collab_tier", "none")
        try:
            condition = max(1, min(5, int(data.get("condition", 3) or 3)))
        except (ValueError, TypeError):
            condition = 3

        row = {
            # Era
            "is_80s":    int(era == "80s"),
            "is_90s":    int(era == "90s"),
            "is_y2k":    int(era == "y2k"),
            "is_current": int(era == "current"),
            # Size
            "size_s":   int(size == "S"),
            "size_m":   int(size == "M"),
            "size_l":   int(size == "L"),
            "size_xl":  int(size == "XL"),
            "size_2xl": int(size == "2XL"),
            # Graphic
            "graphic_snake":      int(graphic == "snake"),
            "graphic_skull":      int(graphic == "skull"),
            "graphic_devil_babe": int(graphic == "devil_babe"),
            "graphic_pin_up":     int(graphic == "pin_up"),
            "graphic_marilyn":    int(graphic == "marilyn"),
            "graphic_cross":      int(graphic == "cross"),
            "graphic_logo_only":  int(graphic == "logo_only"),
            "graphic_other":      int(graphic == "other"),
            # Collab
            "is_supreme_collab": int(collab_tier == "supreme"),
            "is_guess_collab":   int(collab_tier == "guess"),
            "is_other_collab":   int(collab_tier == "other"),
            # Tag / authenticity
            "is_japan_domestic": int(data.get("is_japan_domestic", "0") == "1"),
            "is_reprint":        int(data.get("is_reprint", "0") == "1"),
            # Physical
            "has_back_graphic":  int(data.get("has_back_graphic", "0") == "1"),
            "has_single_stitch": int(data.get("has_single_stitch", "0") == "1"),
            # Condition
            "condition": condition,
        }

    else:
        raise ValueError(f"Unknown brand: {brand}")

    print(f"[RPM row] brand={brand} row={row}", flush=True)
    input_df = pd.DataFrame([row])
    # Align columns to XGBoost training schema — skip for brands using rules-based pricing
    if brand in _models and brand != "hysteric":
        input_df = input_df.reindex(columns=_models[brand]["columns"], fill_value=0)
    return input_df


def predict_for_brand(brand, input_df):
    # Hysteric uses rules-based pricing until new training data is scraped
    if brand == "hysteric":
        return hysteric_rules_predict(input_df.iloc[0].to_dict())
    if brand in _models:
        pred = np.expm1(_models[brand]["model"].predict(input_df)[0])
        return round(float(pred * 0.85), 2), round(float(pred * 1.15), 2)
    return None, None


def normalize_features(raw: dict, brand: str) -> dict:
    """Normalize a raw GPT-4o response against the brand's signal schema.

    Every expected ai_key is guaranteed to be present in the output.
    Missing or null values get explicit type-safe defaults — bool → False,
    select → first option value.  Unexpected keys from the AI are dropped.
    """
    signals = BRANDS[brand]["signals"]
    expected_keys = {sig["ai_key"] for sig in signals}
    normalized = {}
    defaulted = []

    for sig in signals:
        key = sig["ai_key"]
        val = raw.get(key)          # None for both absent keys and explicit null
        if val is None:
            val = False if sig["type"] == "bool" else sig["options"][0][0]
            defaulted.append(key)
        normalized[key] = val

    unexpected = [k for k in raw if k not in expected_keys]
    if unexpected:
        print(f"[RPM normalize] brand={brand} unexpected keys (ignored): {unexpected}", flush=True)
    if defaulted:
        print(f"[RPM normalize] brand={brand} defaulted missing/null keys: {defaulted}", flush=True)

    return normalized


# ── Post-extraction validation rules ─────────────────────────────────────────
# Run after normalize_features, before features reach the frontend or pricing.
# Each rule: "if" fires on the feature dict, "then" corrects specific keys,
# "note" is logged when a correction is made. Add dicts to extend per brand.
BRAND_RULES: dict[str, list[dict]] = {
    "harley": [
        {
            "if":   lambda f: f.get("era") in ("y2k", "current"),
            "then": {"has_3d_emblem": False},
            "note": "3D emblem impossible on Y2K/later era",
        },
    ],
}


def apply_rules(features: dict, brand: str) -> dict:
    corrected = dict(features)
    for rule in BRAND_RULES.get(brand, []):
        if rule["if"](corrected):
            for key, val in rule["then"].items():
                if corrected.get(key) != val:
                    print(f"[RPM rules] {rule['note']} — {key}: {corrected[key]!r} → {val!r}", flush=True)
                    corrected[key] = val
    return corrected


def retail_price(ebay_midpoint: float) -> float:
    """Tiered multiplier: eBay market value → Rogue retail tag price."""
    if ebay_midpoint <= 35:
        multiplier = 1.95
    elif ebay_midpoint <= 80:
        multiplier = 1.70
    elif ebay_midpoint <= 150:
        multiplier = 1.45
    else:
        multiplier = 1.30
    # Round to nearest $5 — cleaner price tags
    raw = ebay_midpoint * multiplier
    return round(raw / 5) * 5


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    retail = None
    form_data = {}
    error = None
    selected_brand = "auto"

    if request.method == "POST":
        selected_brand = request.form.get("brand", "harley")
        form_data = request.form

        try:
            if selected_brand not in BRANDS:
                raise ValueError("Select a brand or upload a photo to auto-detect.")
            input_df = form_to_input_df(selected_brand, form_data)
            low, high = predict_for_brand(selected_brand, input_df)
            if low is None:
                error = f"No model trained for {BRANDS[selected_brand]['label']} yet — add a features CSV to enable pricing."
            else:
                # 3D emblem floor — model trained on cleaned data that excluded
                # premium outliers; prevents underpricing genuine high-end 3D tees.
                # Remove when a dedicated 3D emblem training set is available.
                if selected_brand == "harley" and form_data.get("emblem") == "1":
                    _EMBLEM_FLOOR = 65.0  # eBay midpoint floor — tune from comp data
                    if (low + high) / 2 < _EMBLEM_FLOOR:
                        low  = round(_EMBLEM_FLOOR * 0.85, 2)
                        high = round(_EMBLEM_FLOOR * 1.15, 2)
                        print(f"[RPM emblem-floor] midpoint below ${_EMBLEM_FLOOR:.0f} — floored to ${low:.2f}–${high:.2f}", flush=True)
                result = f"${low:.2f} – ${high:.2f}"
                retail = retail_price((low + high) / 2)

            # Save feedback when a photo was used (image_ref present means /analyze ran)
            image_ref = request.form.get("image_ref", "").strip()
            if image_ref:
                signals = BRANDS[selected_brand]["signals"]
                ai_values   = {s["id"]: request.form.get(f"ai_{s['id']}", "") for s in signals}
                user_values = {s["id"]: request.form.get(s["id"], "")           for s in signals}
                save_feedback(selected_brand, image_ref, ai_values, user_values)

        except Exception as e:
            error = str(e)

    brands_json = json.dumps({
        bid: {"signals": cfg["signals"]}
        for bid, cfg in BRANDS.items()
    })

    return render_template(
        "index.html",
        result=result,
        retail=retail,
        form_data=form_data,
        error=error,
        selected_brand=selected_brand,
        brands=BRANDS,
        brands_json=brands_json,
        trained_brands=list(_models.keys()),
    )


@app.route("/detect", methods=["POST"])
def detect():
    """Stage 1 — fast brand identification from front image only."""
    front = request.files.get("front")
    if not front:
        return jsonify({"error": "Front image required"}), 400
    path = "temp_detect.jpg"
    front.save(path)
    try:
        brand = detect_brand({"front": path})
        return jsonify({"detected_brand": brand})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Stage 2 — full feature extraction using all uploaded images.
    Brand must already be known (resolved by /detect or manual selection).
    Called by the form submit interceptor with the complete current image set.
    """
    brand = request.form.get("brand", "")
    if brand not in BRANDS:
        return jsonify({"error": f"Unknown brand '{brand}' — select a brand first."}), 400

    temp_paths = {}
    try:
        for slot in ("front", "tag", "back", "care"):
            f = request.files.get(slot)
            if f:
                path = f"temp_{slot}.jpg"
                f.save(path)
                temp_paths[slot] = path
        if "front" not in temp_paths:
            return jsonify({"error": "Front image is required"}), 400
        raw = extract_features_from_images(temp_paths, brand=brand)
        print(f"[RPM /analyze] brand={brand} images={list(temp_paths.keys())} raw={json.dumps(raw)}", flush=True)
        features = normalize_features(raw, brand)
        features = apply_rules(features, brand)
        features["image_ref"] = compute_image_ref(temp_paths["front"])
        return jsonify(features)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in temp_paths.values():
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    app.run(debug=True)
