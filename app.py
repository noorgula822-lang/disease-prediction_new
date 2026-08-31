import os
import pandas as pd
import joblib
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("disease_prediction_final_model.pkl")


# ============================================================
# HTML WEBSITE
# ============================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>Disease Prediction System</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background: #f2f5f9;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            color: #1f2937;
            margin-bottom: 30px;
        }

        label {
            font-weight: bold;
            display: block;
            margin-top: 15px;
            color: #374151;
        }

        input,
        select {
            width: 100%;
            padding: 12px;
            margin-top: 6px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 15px;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 14px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            text-align: center;
            background: #eef6ff;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>🩺 Disease Prediction System</h1>

    <form method="POST">

        <label>Age</label>
        <input
            type="number"
            name="age"
            min="1"
            max="120"
            required
        >


        <label>Glucose (mg/dL)</label>
        <input
            type="number"
            name="glucose_mg_dl"
            min="1"
            required
        >


        <label>Cholesterol (mg/dL)</label>
        <input
            type="number"
            name="cholesterol_mg_dl"
            min="1"
            required
        >


        <label>Systolic Blood Pressure</label>
        <input
            type="number"
            name="systolic_bp"
            min="1"
            required
        >


        <label>Diastolic Blood Pressure</label>
        <input
            type="number"
            name="diastolic_bp"
            min="1"
            required
        >


        <label>BMI</label>
        <input
            type="number"
            step="0.1"
            name="bmi"
            min="1"
            required
        >


        <label>Heart Rate</label>
        <input
            type="number"
            name="heart_rate"
            min="1"
            required
        >


        <label>Gender</label>

        <select name="gender">

            <option value="Male">
                Male
            </option>

            <option value="Female">
                Female
            </option>

        </select>


        <label>Smoking</label>

        <select name="smoking">

            <option value="No">
                No
            </option>

            <option value="Yes">
                Yes
            </option>

        </select>


        <label>Alcohol Consumption</label>

        <select name="alcohol_consumption">

            <option value="No">
                No
            </option>

            <option value="Yes">
                Yes
            </option>

        </select>


        <label>Physical Activity</label>

        <select name="physical_activity">

            <option value="Low">
                Low
            </option>

            <option value="Medium">
                Medium
            </option>

            <option value="High">
                High
            </option>

        </select>


        <label>Family History</label>

        <select name="family_history">

            <option value="No">
                No
            </option>

            <option value="Yes">
                Yes
            </option>

        </select>


        <button type="submit">
            Predict Disease
        </button>

    </form>


    {% if result %}

        <div class="result">

            {{ result }}

        </div>

    {% endif %}


</div>

</body>

</html>
"""


# ============================================================
# HOME / PREDICTION ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        # ----------------------------------------------------
        # CREATE PATIENT DATAFRAME
        # ----------------------------------------------------

        patient = pd.DataFrame([{

            "age": float(
                request.form["age"]
            ),

            "glucose_mg_dl": float(
                request.form["glucose_mg_dl"]
            ),

            "cholesterol_mg_dl": float(
                request.form["cholesterol_mg_dl"]
            ),

            "systolic_bp": float(
                request.form["systolic_bp"]
            ),

            "diastolic_bp": float(
                request.form["diastolic_bp"]
            ),

            "bmi": float(
                request.form["bmi"]
            ),

            "heart_rate": float(
                request.form["heart_rate"]
            ),

            "gender": request.form["gender"],

            "smoking": request.form["smoking"],

            "alcohol_consumption":
                request.form["alcohol_consumption"],

            "physical_activity":
                request.form["physical_activity"],

            "family_history":
                request.form["family_history"]

        }])


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(patient)[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = model.predict_proba(patient)[0][1]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            result = (
                "⚠️ Disease Prediction: YES<br>"
                f"Probability: {probability:.2%}"
            )

        else:

            result = (
                "✅ Disease Prediction: NO<br>"
                f"Probability: {probability:.2%}"
            )


    return render_template_string(
        HTML,
        result=result
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
