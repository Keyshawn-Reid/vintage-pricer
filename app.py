from flask import (
    Flask, render_template, request, jsonify,
    session as flask_session, redirect, url_for, send_file, Response,
)
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import json
import sys
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.vision import extract_features_from_image, extract_features_from_images, detect_brand
from src.hysteric_rules import predict_price as hysteric_rules_predict
from src.brands import BRANDS
from src.feedback import compute_image_ref, save_feedback
from src.sessions_db import (
    init_db, create_session, get_sessions, get_session,
    get_session_items, add_item, update_item, update_session_status,
    get_items_by_ids, count_sessions_today,
)

app = Flask(__name__)

# ── Secret key — must be set in environment; no insecure fallback ────────────
_secret = os.environ.get("FLASK_SECRET_KEY")
if not _secret:
    raise RuntimeError(
        "FLASK_SECRET_KEY is not set. "
        "Add it to .env or Render env vars. "
        "Generate one with: python3 -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.secret_key = _secret

# ── Upload limits ─────────────────────────────────────────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB hard cap

SESSION_IMAGES_DIR = os.path.join("data", "session_images")
os.makedirs(SESSION_IMAGES_DIR, exist_ok=True)

# ── Allowed upload types ──────────────────────────────────────────────────────
_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
_ALLOWED_MIMETYPES  = {"image/jpeg", "image/png", "image/webp"}


def _validate_image(file_storage):
    """Return an error string if the upload is invalid, else None."""
    if not file_storage or not file_storage.filename:
        return "No file provided."
    ext = file_storage.filename.rsplit(".", 1)[-1].lower() if "." in file_storage.filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        return f"File type '.{ext}' not allowed — upload JPG, PNG, or WebP."
    mime = (file_storage.content_type or "").split(";")[0].strip().lower()
    if mime and mime not in _ALLOWED_MIMETYPES:
        return f"Unrecognised MIME type '{mime}' — upload JPG, PNG, or WebP."
    return None


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Image too large — 20 MB maximum per file."}), 413

# ── Initialize sessions DB ───────────────────────────────────────────────────
init_db()

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


# ── Template context injected on every request ───────────────────────────────
@app.context_processor
def inject_globals():
    try:
        all_sessions = get_sessions()
    except Exception:
        all_sessions = []

    active_sessions = [s for s in all_sessions if s["status"] == "active"]
    active_session_id = flask_session.get("active_session_id")
    active_session_name = None
    if active_session_id:
        for s in active_sessions:
            if s["id"] == active_session_id:
                active_session_name = s["name"]
                break

    return {
        "active_sessions": active_sessions,
        "active_session_id": active_session_id,
        "active_session_name": active_session_name,
        "summarize_features": summarize_features,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────
def summarize_features(brand: str, features: dict) -> list:
    """Return display strings for notable features (for session item cards)."""
    parts = []
    if brand not in BRANDS:
        return parts
    for sig in BRANDS[brand]["signals"]:
        fid = sig["id"]
        val = str(features.get(fid, ""))
        if not val:
            continue
        if sig["type"] == "bool":
            if val == "1":
                parts.append(sig["label"])
        elif sig["type"] == "select":
            if val and val not in ("unknown", ""):
                for opt_val, opt_label in sig["options"]:
                    if str(opt_val) == val:
                        parts.append(opt_label)
                        break
    return parts


def make_item_title(brand: str, features: dict) -> str:
    if brand not in BRANDS:
        return "Unknown Item"
    brand_label = BRANDS[brand]["label"]
    parts = []
    for sig in BRANDS[brand]["signals"]:
        fid = sig["id"]
        val = str(features.get(fid, ""))
        if sig["type"] == "select" and fid in ("era", "size"):
            if val and val not in ("unknown", ""):
                parts.append(val.upper() if fid == "size" else val)
    if parts:
        return f"{brand_label} · {' · '.join(parts)}"
    return brand_label


def auto_session_name() -> str:
    today = datetime.now(timezone.utc).strftime("%b %-d")
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = count_sessions_today(date_prefix) + 1
    return f"{today} – Run {n}"


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
            "is_80s":    int(era == "80s"),
            "is_90s":    int(era == "90s"),
            "is_y2k":    int(era == "y2k"),
            "is_current": int(era == "current"),
            "size_s":   int(size == "S"),
            "size_m":   int(size == "M"),
            "size_l":   int(size == "L"),
            "size_xl":  int(size == "XL"),
            "size_2xl": int(size == "2XL"),
            "graphic_snake":      int(graphic == "snake"),
            "graphic_skull":      int(graphic == "skull"),
            "graphic_devil_babe": int(graphic == "devil_babe"),
            "graphic_pin_up":     int(graphic == "pin_up"),
            "graphic_marilyn":    int(graphic == "marilyn"),
            "graphic_cross":      int(graphic == "cross"),
            "graphic_logo_only":  int(graphic == "logo_only"),
            "graphic_other":      int(graphic == "other"),
            "is_supreme_collab": int(collab_tier == "supreme"),
            "is_guess_collab":   int(collab_tier == "guess"),
            "is_other_collab":   int(collab_tier == "other"),
            "is_japan_domestic": int(data.get("is_japan_domestic", "0") == "1"),
            "is_reprint":        int(data.get("is_reprint", "0") == "1"),
            "has_back_graphic":  int(data.get("has_back_graphic", "0") == "1"),
            "has_single_stitch": int(data.get("has_single_stitch", "0") == "1"),
            "condition": condition,
        }

    else:
        raise ValueError(f"Unknown brand: {brand}")

    print(f"[RPM row] brand={brand} row={row}", flush=True)
    input_df = pd.DataFrame([row])
    if brand in _models and brand != "hysteric":
        input_df = input_df.reindex(columns=_models[brand]["columns"], fill_value=0)
    return input_df


def predict_for_brand(brand, input_df):
    if brand == "hysteric":
        return hysteric_rules_predict(input_df.iloc[0].to_dict())
    if brand in _models:
        pred = np.expm1(_models[brand]["model"].predict(input_df)[0])
        return round(float(pred * 0.85), 2), round(float(pred * 1.15), 2)
    return None, None


def normalize_features(raw: dict, brand: str) -> dict:
    signals = BRANDS[brand]["signals"]
    expected_keys = {sig["ai_key"] for sig in signals}
    normalized = {}
    defaulted = []

    for sig in signals:
        key = sig["ai_key"]
        val = raw.get(key)
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


BRAND_RULES: dict = {
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
    if ebay_midpoint <= 35:
        multiplier = 1.95
    elif ebay_midpoint <= 80:
        multiplier = 1.70
    elif ebay_midpoint <= 150:
        multiplier = 1.45
    else:
        multiplier = 1.30
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
    low_price = None
    high_price = None
    features_json = None
    image_ref_for_save = None

    if request.method == "POST":
        selected_brand = request.form.get("brand", "harley")
        form_data = request.form

        try:
            if selected_brand not in BRANDS:
                raise ValueError("Select a brand or upload a photo to auto-detect.")
            input_df = form_to_input_df(selected_brand, form_data)
            low, high = predict_for_brand(selected_brand, input_df)
            if low is None:
                error = f"No model trained for {BRANDS[selected_brand]['label']} yet."
            else:
                if selected_brand == "harley" and form_data.get("emblem") == "1":
                    _condition    = form_data.get("condition")
                    _EMBLEM_FLOOR = 70.0 if _condition == "poor" else 90.0
                    _orig_mid     = round((low + high) / 2, 2)
                    if _orig_mid < _EMBLEM_FLOOR:
                        low  = round(_EMBLEM_FLOOR * 0.85, 2)
                        high = round(_EMBLEM_FLOOR * 1.15, 2)
                        print(f"[RPM emblem-floor] condition={_condition!r} floor=${_EMBLEM_FLOOR:.0f} orig_mid=${_orig_mid:.2f} → ${low:.2f}–${high:.2f}", flush=True)
                result = f"${low:.2f} – ${high:.2f}"
                retail = retail_price((low + high) / 2)

                low_price = low
                high_price = high
                image_ref_for_save = request.form.get("image_ref", "").strip()
                features_for_save = {
                    sig["id"]: form_data.get(sig["id"], "")
                    for sig in BRANDS[selected_brand]["signals"]
                }
                features_json = json.dumps(features_for_save)

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
        low_price=low_price,
        high_price=high_price,
        features_json=features_json,
        image_ref_for_save=image_ref_for_save,
    )


@app.route("/detect", methods=["POST"])
def detect():
    front = request.files.get("front")
    if not front:
        return jsonify({"error": "Front image required"}), 400
    err = _validate_image(front)
    if err:
        return jsonify({"error": err}), 400

    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        front.save(path)
        brand = detect_brand({"front": path})
        return jsonify({"detected_brand": brand})
    except Exception as e:
        print(f"[RPM /detect] error: {e}", flush=True)
        return jsonify({"error": "Brand detection failed — try again or select manually."}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route("/analyze", methods=["POST"])
def analyze():
    brand = request.form.get("brand", "")
    if brand not in BRANDS:
        return jsonify({"error": f"Unknown brand '{brand}' — select a brand first."}), 400

    temp_paths = {}
    try:
        for slot in ("front", "tag", "back", "care"):
            f = request.files.get(slot)
            if not f:
                continue
            err = _validate_image(f)
            if err:
                return jsonify({"error": f"Slot '{slot}': {err}"}), 400
            fd, path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            f.save(path)
            temp_paths[slot] = path

        if "front" not in temp_paths:
            return jsonify({"error": "Front image is required"}), 400

        raw = extract_features_from_images(temp_paths, brand=brand)
        print(f"[RPM /analyze] brand={brand} images={list(temp_paths.keys())} raw={json.dumps(raw)}", flush=True)
        features = normalize_features(raw, brand)
        features = apply_rules(features, brand)
        image_ref = compute_image_ref(temp_paths["front"])
        features["image_ref"] = image_ref

        img_save_path = os.path.join(SESSION_IMAGES_DIR, f"{image_ref}.jpg")
        if not os.path.exists(img_save_path):
            shutil.copy2(temp_paths["front"], img_save_path)

        return jsonify(features)
    except Exception as e:
        print(f"[RPM /analyze] error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    finally:
        for p in temp_paths.values():
            if os.path.exists(p):
                os.remove(p)


# ── Session routes ────────────────────────────────────────────────────────────
@app.route("/sessions")
def sessions_list():
    sessions = get_sessions()
    return render_template("sessions.html", sessions=sessions)


@app.route("/sessions/new", methods=["POST"])
def sessions_new():
    name = request.form.get("name", "").strip() or auto_session_name()
    session_id = create_session(name)
    flask_session["active_session_id"] = session_id
    return redirect(url_for("session_detail", session_id=session_id))


@app.route("/sessions/<session_id>")
def session_detail(session_id):
    sess = get_session(session_id)
    if not sess:
        return redirect(url_for("sessions_list"))
    items = get_session_items(session_id)
    # Viewing a session makes it the active session
    flask_session["active_session_id"] = session_id
    return render_template(
        "session_detail.html",
        session=sess,
        items=items,
        brands=BRANDS,
    )


@app.route("/sessions/save-item", methods=["POST"])
def save_item():
    session_id = request.form.get("session_id", "").strip()
    brand = request.form.get("brand", "").strip()
    image_ref = request.form.get("image_ref", "").strip()
    features_json_str = request.form.get("features_json", "{}")
    low_price = request.form.get("low_price", "0")
    high_price = request.form.get("high_price", "0")

    try:
        features = json.loads(features_json_str)
    except Exception:
        features = {}

    try:
        low = float(low_price)
        high = float(high_price)
    except (ValueError, TypeError):
        low = high = 0.0

    sess = get_session(session_id)
    if not sess:
        return redirect("/?save_error=1")

    title = make_item_title(brand, features)
    rogue_retail = retail_price((low + high) / 2) if low or high else None
    add_item(
        session_id=session_id,
        brand=brand,
        title=title,
        category="T-Shirt",
        features=features,
        image_ref=image_ref,
        suggested_low=low,
        suggested_high=high,
        final_price=float(rogue_retail) if rogue_retail else None,
    )
    flask_session["active_session_id"] = session_id
    return redirect("/?saved=1")


@app.route("/sessions/<session_id>/items/<item_id>/update", methods=["POST"])
def update_session_item(session_id, item_id):
    try:
        data = request.get_json(silent=True) or {}
        final_price = data.get("final_price")
        status = data.get("status")
        update_item(item_id, final_price=final_price, status=status)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sessions/<session_id>/status", methods=["POST"])
def update_session_status_route(session_id):
    try:
        data = request.get_json(silent=True) or {}
        status = data.get("status", "")
        if status in ("active", "submitted", "printed"):
            update_session_status(session_id, status)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sessions/<session_id>/export")
def export_session(session_id):
    item_ids_param = request.args.get("items", "")
    item_ids = [i.strip() for i in item_ids_param.split(",") if i.strip()]

    items = get_items_by_ids(item_ids) if item_ids else get_session_items(session_id)

    data = []
    for item in items:
        data.append({
            "id": item["id"],
            "brand": BRANDS.get(item["brand"], {}).get("label", item["brand"]),
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "features": item.get("features", {}),
            "suggested_range": (
                f"${item['suggested_low']:.2f} – ${item['suggested_high']:.2f}"
                if item.get("suggested_low") is not None else ""
            ),
            "final_price": item.get("final_price"),
            "status": item.get("status", "pending"),
        })

    payload = json.dumps(data, indent=2)
    return Response(
        payload,
        status=200,
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}.json"},
    )


@app.route("/session_images/<image_ref>")
def serve_session_image(image_ref):
    if not re.match(r"^[a-f0-9]{16}$", image_ref):
        return "", 404
    path = os.path.join(SESSION_IMAGES_DIR, f"{image_ref}.jpg")
    if not os.path.exists(path):
        return "", 404
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True)
