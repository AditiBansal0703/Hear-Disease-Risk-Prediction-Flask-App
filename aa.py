import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Example input (replace with actual values)
input_data = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 48, 1]
features = np.array([input_data])

# Get prediction and probability
prediction_prob = model.predict_proba(features)[:, 1]
print(f"Prediction Probability: {prediction_prob}")

prediction = 1 if prediction_prob > 0.5 else 0
print(f"Prediction: {prediction}")
