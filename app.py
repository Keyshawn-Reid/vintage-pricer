from flask import Flask, render_template, request
import pandas as pd
import sys
import os
from src.features import extract_features
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

# Train model on startup
df = pd.read_csv("data/raw/harley_features.csv")
X = df.drop("sold_price", axis=1)
y = df["sold_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

def predict_price(title):
    features = extract_features(title)
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    return round(float(prediction * 0.85), 2), round(float(prediction * 1.15), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        title = request.form["title"]
        low, high = predict_price(title)
        result = f"${low:.2f} - ${high:.2f}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)