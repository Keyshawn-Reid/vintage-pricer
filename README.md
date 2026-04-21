# RPM — Rogue Pricing Model

AI-powered vintage clothing pricer for in-store use. Upload a photo, the model extracts signals, and returns a price range.

🚀 **Live:** https://vintage-pricer.onrender.com

---

## Current Status — v1.2

- Single unified flow: photo upload → AI feature extraction → user review → price
- Three supported brands: Harley Davidson, Ed Hardy, Hysteric Glamour
- Per-brand XGBoost models trained on startup from CSV data
- GPT-4o vision extracts brand-specific signals from uploaded photos
- Brand-adaptive signal fields — only relevant features appear per brand
- Deployed on Render via Gunicorn

**Data state:**
- Harley Davidson: 114 real eBay sold listings (model active, accuracy limited — see weaknesses)
- Ed Hardy / Hysteric Glamour: synthetic placeholder data — models load but prices are not market-accurate pending real CSV data

---

## How It Works

1. User uploads a photo (tag, front, or back)
2. Image is compressed via Pillow and sent to GPT-4o Vision
3. GPT-4o returns a structured JSON feature dict (era, size, brand-specific signals)
4. Brand field and signal dropdowns are auto-filled
5. User reviews and adjusts if needed
6. XGBoost model predicts a price; output is `prediction × [0.85, 1.15]`

---

## Architecture

**Single-file Flask app** (`app.py`) — trains all brand models on startup, serves two routes:
- `GET/POST /` — renders the pricing UI, handles form submission
- `POST /analyze` — accepts image + brand, returns extracted features as JSON

**Data pipeline per brand:**
1. `data/raw/{brand}_raw.csv` — title + sold_price (manually scraped)
2. `src/features.py` — regex/keyword extractor → writes `{brand}_features.csv`
3. `src/brands.py` — brand config: signal schemas, vision prompts, feature column definitions
4. `app.py` — trains XGBRegressor per brand on startup, routes predictions

**Feature dict schema (Harley):**
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

Each brand has its own schema defined in `src/brands.py`.

---

## Strengths

- **Domain-specific use case** — built for in-store vintage resellers making real buy decisions, not a generic price predictor
- **LLM + classical ML pipeline** — GPT-4o extracts structured features from unstructured images; XGBoost handles the regression. Clean separation of concerns
- **Brand-adaptive signals** — signal fields and vision prompts are different per brand; the system understands that "single stitch" matters for Harley but not Ed Hardy
- **Deployed and iterating** — live product with real user feedback loop, not just a notebook
- **Editorial UI** — designed for mobile in-store use, not a dashboard

---

## Areas for Improvement

**Model accuracy (critical)**
The Harley model's cross-validated MAE ($45.41) is comparable to simply predicting the mean price ($44.21). The model needs more data (target: 400–500 rows), log-transformation of the price target to compress outliers, and removal of `is_vtg` (53% feature importance at 96% prevalence — it's noise). Until fixed, the ±15% band is not a real confidence interval.

**Real data for Ed Hardy and Hysteric Glamour**
Current models run on synthetic data. Prices are not market-accurate. Replace CSVs with real scraped sold listings before relying on these models.

**Model persistence**
Models retrain on every cold start. Add `joblib.dump/load` to serialize trained models as artifacts. This also enables versioning and faster startup.

**Concurrency — temp file race condition**
`temp_analyze.jpg` is a hardcoded shared path. Concurrent uploads will overwrite each other. Fix: use `tempfile.NamedTemporaryFile` or a UUID-named path per request.

**No test coverage**
Zero assertions in the codebase. At minimum, add unit tests for `form_to_input_df`, `predict_for_brand`, and `extract_features` to catch regressions when adding new brands.

**Missing `.gitignore`**
Test images are committed to the repo. `.env` is unprotected. Add a standard Python `.gitignore`.

---

## Commands

```bash
# Run the Flask dev server
python app.py

# Run the production server (as deployed on Render)
gunicorn app:app

# Regenerate features CSV from raw eBay listings (Harley)
python src/features.py

# Test vision extraction on an image
python src/vision.py
```

**Environment variables** (in `.env`):
- `OPENAI_API_KEY` — required for GPT-4o image analysis
- `EBAY_APP_ID` — eBay API (not yet active; data is manually scraped)

---

## Roadmap

**Next (data):**
- Expand Harley dataset to 400+ rows, fix model accuracy
- Scrape and add real Ed Hardy and Hysteric Glamour sold listings
- Add per-brand feature extractor scripts (`src/features_ed_hardy.py`, etc.)

**Next (product):**
- Comp cards: show 3 most similar sold listings alongside the price estimate
- Feedback capture: "what did it actually sell for?" input to grow the dataset
- Model persistence with `joblib` — serialize trained models, version them

**Later:**
- Generics / non-brand pricing track
- Collab/hype item detection
- Confidence score derived from actual model error, not a fixed ±15%
