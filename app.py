from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

from database import (
    create_database,
    save_prediction,
    get_predictions,
    get_prediction_summary,
    get_crop_summary,
    delete_prediction,
    search_predictions
)

app = Flask(__name__)

# Load final trained ML model
model = joblib.load("crop_yield_model_compressed.pkl")

# Create database and predictions table
create_database()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            year = int(request.form["year"])
            area = float(request.form["area"])
            rainfall = float(request.form["rainfall"])
            fertilizer = float(request.form["fertilizer"])
            pesticide = float(request.form["pesticide"])

            state = request.form["state"].strip()
            crop = request.form["crop"].strip()
            season = request.form["season"].strip()

            # Server-side validation
            if year < 2000 or year > 2026:
                raise ValueError("Year must be between 2000 and 2026.")

            if area < 0:
                raise ValueError("Area cannot be negative.")

            if rainfall < 0:
                raise ValueError("Annual rainfall cannot be negative.")

            if fertilizer < 0:
                raise ValueError("Fertilizer cannot be negative.")

            if pesticide < 0:
                raise ValueError("Pesticide cannot be negative.")

            if not state:
                raise ValueError("Please enter a state.")

            if not crop:
                raise ValueError("Please enter a crop.")

            if not season:
                raise ValueError("Please select a season.")

            # Prepare input data
            new_data = pd.DataFrame([{
                "Year": year,
                "State": state,
                "Crop": crop,
                "Season": season,
                "Area": area,
                "Annual_Rainfall": rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide
            }])

            # Generate prediction
            prediction_log = model.predict(new_data)
            prediction = np.expm1(prediction_log[0])

            # Save prediction to database
            save_prediction(
                year=year,
                state=state,
                crop=crop,
                season=season,
                area=area,
                rainfall=rainfall,
                fertilizer=fertilizer,
                pesticide=pesticide,
                predicted_yield=float(prediction)
            )

        except ValueError as e:

            error = str(e)

        except Exception as e:

            error = "Unable to generate prediction. Please check your input values."

            print("Error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )
@app.route("/history")
def history():
    search_text = request.args.get("search", "").strip()
    selected_season = request.args.get("season", "").strip()

    predictions = search_predictions(
        search_text,
        selected_season
    )

    return render_template(
        "history.html",
        predictions=predictions,
        search_text=search_text,
        selected_season=selected_season
    )


@app.route("/dashboard")
def dashboard():

    summary = get_prediction_summary()
    crop_summary = get_crop_summary()

    # Calculate bar width percentage for chart
    if crop_summary:
        max_count = max(crop[1] for crop in crop_summary)

        crop_summary = [
            (
                crop[0],
                crop[1],
                crop[2],
                (crop[1] / max_count) * 100 if max_count > 0 else 0
            )
            for crop in crop_summary
        ]

    return render_template(
        "dashboard.html",
        summary=summary,
        crop_summary=crop_summary
    )

@app.route("/model-performance")
def model_performance():

    metrics = pd.read_csv("model_metrics.csv")

    metrics = metrics.to_dict("records")

    return render_template(
        "model_performance.html",
        metrics=metrics
    )

@app.route("/delete/<int:prediction_id>", methods=["POST"])
def delete_prediction_route(prediction_id):

    delete_prediction(prediction_id)

    return render_template(
        "history.html",
        predictions=get_predictions()
    )


if __name__ == "__main__":
    app.run(debug=True)