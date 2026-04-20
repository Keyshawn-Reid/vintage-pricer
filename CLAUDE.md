# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the Flask dev server
python app.py

# Run the production server (as deployed on Render)
gunicorn app:app

# Regenerate features CSV from raw eBay listings
python src/features.py

# Test the model via CLI (interactive title input)
python src/model.py

# Test vision extraction on an image
python src/vision.py
```

Environment variables required (in `.env`):
- `OPENAI_API_KEY` — used by `src/vision.py` for GPT-4o image analysis
- `EBAY_APP_ID` — used by `src/scraper.py` (eBay API not yet active)

## Architecture

The app is a single-file Flask app (`app.py`) that trains an XGBoost model on startup from `data/raw/harley_features.csv`. There is no model persistence — it retrains every cold start.

**Data pipeline:**
1. `data/raw/harley_raw.csv` — raw eBay sold listings (title + sold_price)
2. `src/features.py` — parses titles via regex into binary feature columns, writes `harley_features.csv`
3. `src/model.py` — loads `harley_features.csv`, trains XGBoost, exposes `predict_price(title)` and `predict_price_from_features(features_dict)`

**Two pricing paths in `app.py`:**
- **Manual form** (`mode=manual`): user selects era, size, emblem, stitching, etc. via dropdowns → `predict_from_form()` maps form strings to the feature dict schema → `predict_from_features()`
- **Photo** (`mode=photo`): image uploaded → `src/vision.py` sends it to GPT-4o → returns JSON matching the same feature dict schema → same `predict_from_features()` call

Price output is always `(prediction * 0.85, prediction * 1.15)` — a ±15% band around the raw model prediction.

**Feature dict schema** (shared contract between vision.py, model.py, and app.py):
```python
{
    "era": "80s" | "90s" | "y2k" | "unknown",
    "size": "S" | "M" | "L" | "XL" | "2XL" | "unknown",
    "has_3d_emblem": bool,
    "has_single_stitch": bool,
    "has_location_name": bool,
    "is_event_tee": bool
}
```

**Note:** `src/vision.py` uses `Path(__file__).resolve().parent.parent` to dynamically locate the project root, so it works on any machine.

## Deployment

Deployed on Render via `Procfile` (`gunicorn app:app`). Live at https://vintage-pricer.onrender.com. The model retrains on every deploy since there is no model serialization.

## Roadmap Context

Current brand scope is Harley Davidson only. The roadmap (v1.1–v1.5) targets a unified photo+form flow, multi-brand support (Ed Hardy, Hysteric Glamour), per-brand XGBoost models, and mobile UX. New brands will each need their own raw CSV, feature extractor, and trained model following the existing Harley pattern.
