from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Train model on startup
df = pd.read_csv("data/raw/harley_features.csv")
X = df.drop("sold_price", axis=1)
y = df["sold_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

def predict_from_form(data):
    features = {
        "is_80s": 1 if data["era"] == "80s" else 0,
        "is_90s": 1 if data["era"] == "90s" else 0,
        "is_y2k": 1 if data["era"] == "y2k" else 0,
        "has_3d_emblem": int(data["emblem"]),
        "has_single_stitch": int(data["single_stitch"]),
        "is_vtg": 1,
        "size_s": 1 if data["size"] == "S" else 0,
        "size_m": 1 if data["size"] == "M" else 0,
        "size_l": 1 if data["size"] == "L" else 0,
        "size_xl": 1 if data["size"] == "XL" else 0,
        "has_location": int(data["location"]),
        "is_event_tee": int(data["event"]),
        "has_year": 0
    }
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    return round(float(prediction * 0.85), 2), round(float(prediction * 1.15), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}
    if request.method == "POST":
        form_data = request.form
        low, high = predict_from_form(form_data)
        result = f"${low:.2f} - ${high:.2f}"
    return render_template("index.html", result=result, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)