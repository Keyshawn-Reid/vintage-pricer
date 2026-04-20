from flask import Flask, render_template, request, jsonify
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.vision import extract_features_from_image
from src.brands import BRANDS

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
    _y = _df["sold_price"]
    _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2, random_state=42)
    _m = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    _m.fit(_X_train, _y_train)
    _models[_brand_id] = {"model": _m, "columns": list(_X.columns)}
    print(f"[{_brand_id}] model trained on {len(_df)} rows")


# ── Feature extraction: form → input DataFrame ──────────────────────────────
def form_to_input_df(brand, data):
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
        row = {
            "is_80s": int(era == "80s"),
            "is_90s": int(era == "90s"),
            "is_y2k": int(era == "y2k"),
            "size_s": int(size == "S"),
            "size_m": int(size == "M"),
            "size_l": int(size == "L"),
            "size_xl": int(size == "XL"),
            "graphic_devil": int(graphic == "devil"),
            "graphic_pin_up": int(graphic == "pin_up"),
            "graphic_logo": int(graphic == "logo"),
            "is_japan_market": int(data.get("japan_market", "0") == "1"),
            "is_collab": int(data.get("collab", "0") == "1"),
            "is_vtg": 1,
        }

    else:
        raise ValueError(f"Unknown brand: {brand}")

    input_df = pd.DataFrame([row])
    # Align columns to training schema in case of any mismatch
    if brand in _models:
        input_df = input_df.reindex(columns=_models[brand]["columns"], fill_value=0)
    return input_df


def predict_for_brand(brand, input_df):
    if brand not in _models:
        return None, None
    pred = _models[brand]["model"].predict(input_df)[0]
    return round(float(pred * 0.85), 2), round(float(pred * 1.15), 2)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}
    error = None
    selected_brand = "harley"

    if request.method == "POST":
        selected_brand = request.form.get("brand", "harley")
        form_data = request.form

        try:
            input_df = form_to_input_df(selected_brand, form_data)
            low, high = predict_for_brand(selected_brand, input_df)
            if low is None:
                error = f"No model trained for {BRANDS[selected_brand]['label']} yet — add a features CSV to enable pricing."
            else:
                result = f"${low:.2f} – ${high:.2f}"
        except Exception as e:
            error = str(e)

    brands_json = json.dumps({
        bid: {"signals": cfg["signals"]}
        for bid, cfg in BRANDS.items()
    })

    return render_template(
        "index.html",
        result=result,
        form_data=form_data,
        error=error,
        selected_brand=selected_brand,
        brands=BRANDS,
        brands_json=brands_json,
        trained_brands=list(_models.keys()),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"error": "No photo provided"}), 400
    brand = request.form.get("brand", "harley")
    path = "temp_analyze.jpg"
    photo.save(path)
    try:
        features = extract_features_from_image(path, brand=brand)
        return jsonify(features)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    app.run(debug=True)
