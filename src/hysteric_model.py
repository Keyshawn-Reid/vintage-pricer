import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hysteric_features import extract_features, map_condition

df = pd.read_csv("data/raw/hysteric_features.csv")

X = df.drop("sold_price", axis=1)
y = df["sold_price"]
y_log = np.log1p(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_maes = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y.iloc[val_idx]

    m = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    m.fit(X_tr, y_tr)
    preds = np.expm1(m.predict(X_val))
    fold_maes.append(mean_absolute_error(y_val, preds))

cv_mae       = np.mean(fold_maes)
baseline_mae = mean_absolute_error(y, np.full(len(y), y.median()))
print(f"CV MAE:       ${cv_mae:.2f}  (avg over 5 folds)")
print(f"Baseline MAE: ${baseline_mae:.2f}  (median predictor)")
print(f"Gap:          ${baseline_mae - cv_mae:.2f}")

# Feature importance
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y_log)

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n--- Feature importance (top 10) ---")
print(importance.sort_values(ascending=False).head(10).to_string())

FEATURE_COLUMNS = list(X.columns)


def predict_price(title: str) -> tuple[float, float]:
    features = extract_features(title)
    input_df = pd.DataFrame([features]).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    prediction = np.expm1(model.predict(input_df)[0])
    return round(prediction * 0.85, 2), round(prediction * 1.15, 2)


def predict_price_from_features(features: dict) -> tuple[float, float]:
    """
    features dict keys:
      era           — "80s" | "90s" | "y2k" | "unknown"
      category      — "tee" | "hoodie" | "jacket" | "denim" | "knitwear" | "shirt"
      is_supreme_collab  — bool
      is_band_collab     — bool
      is_collab          — bool
      is_made_in_japan   — bool
      has_skull_graphic  — bool
      has_girl_graphic   — bool
      has_logo_print     — bool
      size          — "XS" | "S" | "M" | "L" | "XL" | "unknown"
      condition     — int 1–5
    """
    era = features.get("era", "unknown")
    cat = features.get("category", "")
    size = features.get("size", "unknown")

    input_data = {
        "is_80s":             1 if era == "80s" else 0,
        "is_90s":             1 if era == "90s" else 0,
        "is_y2k":             1 if era == "y2k" else 0,
        "is_vintage":         1 if era != "unknown" else 0,
        "is_supreme_collab":  int(features.get("is_supreme_collab", False)),
        "is_band_collab":     int(features.get("is_band_collab", False)),
        "is_collab":          int(features.get("is_collab", False)),
        "is_made_in_japan":   int(features.get("is_made_in_japan", False)),
        "cat_tee":            1 if cat == "tee" else 0,
        "cat_hoodie":         1 if cat == "hoodie" else 0,
        "cat_jacket":         1 if cat == "jacket" else 0,
        "cat_denim":          1 if cat == "denim" else 0,
        "cat_knitwear":       1 if cat == "knitwear" else 0,
        "cat_shirt":          1 if cat == "shirt" else 0,
        "has_skull_graphic":  int(features.get("has_skull_graphic", False)),
        "has_girl_graphic":   int(features.get("has_girl_graphic", False)),
        "has_logo_print":     int(features.get("has_logo_print", False)),
        "condition":          int(features.get("condition", 3)),
        "size_xs":            1 if size == "XS" else 0,
        "size_s":             1 if size == "S" else 0,
        "size_m":             1 if size == "M" else 0,
        "size_l":             1 if size == "L" else 0,
        "size_xl":            1 if size in ("XL", "2XL", "3XL") else 0,
    }

    input_df = pd.DataFrame([input_data]).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    prediction = np.expm1(model.predict(input_df)[0])
    return round(float(prediction * 0.85), 2), round(float(prediction * 1.15), 2)


if __name__ == "__main__":
    print("\n--- Hysteric Glamour Pricer ---")
    while True:
        title = input("\nEnter item title (or 'quit' to exit): ")
        if title.lower() == "quit":
            break
        low, high = predict_price(title)
        print(f"Estimated price range: ${low} – ${high}")
