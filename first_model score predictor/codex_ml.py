
# codex_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# =========================
# LOAD DATA
# =========================

# Replace with your CSV file name
df = pd.read_csv("/home/vivek2006/Desktop/machine learning/student_ml_dataset_100_rows.csv")

print("First 5 Rows:\n")
print(df.head())

# =========================
# FEATURES AND TARGETS
# =========================

# Input Features
X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignments_Completed",
        "Previous_Score"
    ]
]

# Target for Final Score Prediction
y_score = df["Final_Score"]

# Target for Pass/Fail Prediction
label_encoder = LabelEncoder()
y_pass = label_encoder.fit_transform(df["Passed"])
# Yes -> 1
# No  -> 0

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_score_train, y_score_test = train_test_split(
    X,
    y_score,
    test_size=0.2,
    random_state=42
)

_, _, y_pass_train, y_pass_test = train_test_split(
    X,
    y_pass,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL 1 : FINAL SCORE
# =========================

score_model = LinearRegression()

score_model.fit(X_train, y_score_train)

score_predictions = score_model.predict(X_test)

score_error = mean_absolute_error(y_score_test, score_predictions)

print("\nFinal Score Model Trained")
print("Mean Absolute Error:", score_error)

# =========================
# MODEL 2 : PASS / FAIL
# =========================

pass_model = LogisticRegression()

pass_model.fit(X_train, y_pass_train)

pass_predictions = pass_model.predict(X_test)

pass_accuracy = accuracy_score(y_pass_test, pass_predictions)

print("\nPass/Fail Model Trained")
print("Accuracy:", pass_accuracy)

# =========================
# USER INPUT PREDICTION
# =========================

print("\n===== ENTER STUDENT DATA =====")

study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance: "))
assignments = int(input("Assignments Completed: "))
previous_score = float(input("Previous Score: "))

new_data = [[
    study_hours,
    attendance,
    assignments,
    previous_score
]]

# Predict Final Score
predicted_score = score_model.predict(new_data)[0]

# Predict Pass/Fail
predicted_pass = pass_model.predict(new_data)[0]

pass_result = label_encoder.inverse_transform([predicted_pass])[0]

print("\n===== PREDICTION RESULT =====")
print(f"Predicted Final Score: {predicted_score:.2f}")
print(f"Pass/Fail Prediction: {pass_result}")

# =========================
# SAVE MODELS
# =========================

joblib.dump(score_model, "final_score_model.pkl")
joblib.dump(pass_model, "pass_fail_model.pkl")

print("\nModels saved successfully!")

