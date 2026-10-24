from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model, scaler, and feature order
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read numerical inputs
        input_data = {
            "area": float(request.form["area"]),
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": int(request.form["bathrooms"]),
            "stories": int(request.form["stories"]),
            "parking": int(request.form["parking"]),
        }

        # Convert yes/no dropdowns into one-hot values
        yes_no_fields = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
        ]

        for field in yes_no_fields:
            input_data[f"{field}_yes"] = (
                1 if request.form[field] == "yes" else 0
            )

        # Convert furnishing status into the two encoded columns
        furnishing = request.form["furnishingstatus"]

        input_data["furnishingstatus_semi-furnished"] = (
            1 if furnishing == "semi-furnished" else 0
        )

        input_data["furnishingstatus_unfurnished"] = (
            1 if furnishing == "unfurnished" else 0
        )

        # Create DataFrame and match training feature order
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scale all features, matching the notebook's preprocessing
        scaled_values = scaler.transform(input_df)

        input_df = pd.DataFrame(
            scaled_values,
            columns=feature_columns
        )

        # Predict house price
        predicted_price = float(model.predict(input_df)[0])

        formatted_price = f"{predicted_price:,.2f}"

        return render_template(
            "result.html",
            prediction=formatted_price
        )

    except Exception as error:
        app.logger.exception("Prediction failed")
        return (
            "Could not make the prediction. Error: " + str(error),
            500
        )


if __name__ == "__main__":
    app.run(debug=True)