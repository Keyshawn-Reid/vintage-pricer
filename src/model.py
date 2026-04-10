import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features

df = pd.read_csv("data/raw/harley_features.csv")

X = df.drop("sold_price", axis=1)
y = df["sold_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Model trained. MAE: ${round(mae, 2)}")

def predict_price(title):
    features = extract_features(title)
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    low = round(prediction * 0.85, 2)
    high = round(prediction * 1.15, 2)
    return round(low, 2), round(high, 2)

def predict_price_from_features(features):
    input_data = {
        "is_80s": 1 if features["era"] == "80s" else 0,
        "is_90s": 1 if features["era"] == "90s" else 0,
        "is_y2k": 1 if features["era"] == "y2k" else 0,
        "has_3d_emblem": int(features["has_3d_emblem"]),
        "has_single_stitch": int(features["has_single_stitch"]),
        "is_vtg": 1,
        "size_s": 1 if features["size"] == "S" else 0,
        "size_m": 1 if features["size"] == "M" else 0,
        "size_l": 1 if features["size"] == "L" else 0,
        "size_xl": 1 if features["size"] == "XL" else 0,
        "has_location": int(features["has_location_name"]),
        "is_event_tee": int(features["is_event_tee"]),
        "has_year": 0
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return round(float(prediction * 0.85), 2), round(float(prediction * 1.15), 2)

if __name__ == "__main__":
    print("\n--- Vintage Pricer ---")
    while True:
        title = input("\nEnter item title (or 'quit' to exit): ")
        if title.lower() == "quit":
            break
        low, high = predict_price(title)
        print(f"Estimated price range: ${low} - ${high}")