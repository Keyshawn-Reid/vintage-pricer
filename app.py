from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.vision import extract_features_from_image

app = Flask(__name__)

# Train model on startup
df = pd.read_csv("data/raw/harley_features.csv")
X = df.drop("sold_price", axis=1)
y = df["sold_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

def predict_from_features(features):
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

def predict_from_form(data):
    features = {
        "era": data["era"],
        "has_3d_emblem": data["emblem"] == "1",
        "has_single_stitch": data["single_stitch"] == "1",
        "size": data["size"],
        "has_location_name": data["location"] == "1",
        "is_event_tee": data["event"] == "1"
    }
    return predict_from_features(features)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    extracted = None
    form_data = {}
    error = None

    if request.method == "POST":
        mode = request.form.get("mode")

        if mode == "photo":
            photo = request.files.get("photo")
            if photo:
                path = "temp_upload.jpg"
                photo.save(path)
                try:
                    extracted = extract_features_from_image(path)
                    low, high = predict_from_features(extracted)
                    result = f"${low:.2f} - ${high:.2f}"
                except Exception as e:
                    error = f"Could not process image: {str(e)}"
                finally:
                    if os.path.exists(path):
                        os.remove(path)
        else:
            form_data = request.form
            low, high = predict_from_form(form_data)
            result = f"${low:.2f} - ${high:.2f}"

    return render_template("index.html", result=result, form_data=form_data, extracted=extracted, error=error, mode=request.form.get("mode", "manual"))

if __name__ == "__main__":
    app.run(debug=True)