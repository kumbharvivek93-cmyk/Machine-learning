from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("final_score_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get data from form
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        assignments = int(request.form["assignments"])
        previous_score = float(request.form["previous_score"])

        # Convert to numpy array
        data = np.array([[
            study_hours,
            attendance,
            assignments,
            previous_score
        ]])

        # Predict final score
        prediction = model.predict(data)[0]

        # Pass / Fail Logic
        result = "PASS" if prediction >= 40 else "FAIL"

        return render_template(
            "index.html",
            prediction=round(prediction, 2),
            result=result
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True) 


    
