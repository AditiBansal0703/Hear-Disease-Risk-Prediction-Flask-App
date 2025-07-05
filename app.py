from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect and map form data
            data = []

            # Gender (0 for Female, 1 for Male)
            gender = int(request.form.get('Gender', 0))
            data.append(gender)

            # List of symptoms expected from the form (all binary: 0/1)
            symptoms = [
                'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations',
                'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea',
                'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity',
                'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress'
            ]

            # Read each symptom from form safely
            for symptom in symptoms:
                value = int(request.form.get(symptom, 0))
                data.append(value)

            # Age (numeric)
            age = int(request.form.get('Age', 0))
            data.append(age)

            # Convert to numpy array for model
            features = np.array([data])

            # Predict probability
            prob = model.predict_proba(features)[0][1]  # probability for class 1 (high risk)

            # Set a custom threshold
            threshold = 0.3  # adjust based on what sensitivity you want
            prediction = 1 if prob >= threshold else 0

            # Return the result page
            return render_template('result.html', prediction=prediction, probability=round(prob * 100, 2))

        except Exception as e:
            # If something fails, return an error message
            return f"Something went wrong: {str(e)}", 500  # 500 means "Internal Server Error"

    # If GET request, show the input form
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')

if __name__ == '__main__':
    app.run(debug=True)
