from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
knn = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # IMPORTANT: EXACT SAME ORDER AS TRAINING
        features = [
            float(request.form['StudyTimeWeekly']),
            float(request.form['Absences']),
            float(request.form['Tutoring']),
            float(request.form['ParentalSupport']),
            float(request.form['Extracurricular']),
            float(request.form['Sports']),
            float(request.form['Music']),
            float(request.form['GradeClass']),
            float(request.form['ReadingHabit'])   
        ]

        data = np.array([features])
        data = scaler.transform(data)

        prediction = knn.predict(data)[0]

        return render_template('index.html', result=round(prediction, 2))

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)