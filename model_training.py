import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')

# Check if any columns need to be encoded or transformed
# Example: Encoding gender if it's categorical (Male = 1, Female = 0)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Check the distribution of the Heart_Risk column (data imbalance)
print(data['Heart_Risk'].value_counts())

# Optionally, plot the distribution
data['Heart_Risk'].value_counts().plot(kind='bar')
plt.title("Distribution of Heart Risk (0 = Low Risk, 1 = High Risk)")
plt.show()

# Features and Target
X = data.drop('Heart_Risk', axis=1)
y = data['Heart_Risk']

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

model.fit(X_train, y_train)

# Predict and print accuracy
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importances = model.feature_importances_
feature_names = X.columns

# Display feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
