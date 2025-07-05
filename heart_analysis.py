# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# data = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')

# # 1. Correlation heatmap
# plt.figure(figsize=(12,10))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
# plt.title('Feature Correlation Heatmap')
# plt.savefig('static/heatmap.png')
# plt.close()

# # Target distribution (with hue set to 'Heart_Risk' to avoid warning)
# plt.figure(figsize=(6,4))
# sns.countplot(x='Heart_Risk', data=data, hue='Heart_Risk', palette='Set2', legend=False)
# plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
# plt.savefig('static/target_distribution.png')
# plt.close()


# # 3. Pairplot of features vs target (Visualizing relationships between features and target)
# sns.pairplot(data, hue='Heart_Risk', palette='Set2', height=2)
# plt.savefig('static/pairplot.png')
# plt.close()

# # 4. Distribution of numerical features
# numerical_features = ['Age', 'Chronic_Stress', 'Fatigue', 'Shortness_of_Breath']
# for feature in numerical_features:
#     plt.figure(figsize=(6,4))
#     sns.histplot(data[feature], kde=True, color='blue', bins=20)
#     plt.title(f'Distribution of {feature}')
#     plt.savefig(f'static/{feature}_distribution.png')
#     plt.close()

# # 5. Box plots to see feature distributions by Heart_Risk
# for feature in numerical_features:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(x='Heart_Risk', y=feature, data=data, palette='Set2')
#     plt.title(f'{feature} Distribution by Heart Risk')
#     plt.savefig(f'static/{feature}_boxplot.png')
#     plt.close()

# # 6. Bar plots for categorical features
# categorical_features = ['Gender', 'Smoking', 'Family_History']
# for feature in categorical_features:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x=feature, hue='Heart_Risk', data=data, palette='Set2')
#     plt.title(f'{feature} Count by Heart Risk')
#     plt.savefig(f'static/{feature}_countplot.png')
#     plt.close()

# # 7. Pairplot with selected numerical features (Age, Chronic_Stress, Fatigue, etc.)
# sns.pairplot(data[['Age', 'Chronic_Stress', 'Fatigue', 'Heart_Risk']], hue='Heart_Risk', palette='Set2', height=2)
# plt.savefig('static/numerical_pairplot.png')
# plt.close()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')

# Use only a subset of the data (optional, for faster testing)
data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data

# 1. Correlation heatmap (simplified for smaller dataset)
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('static/heatmap.png')
plt.close()

# 2. Target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Heart_Risk', data=data, hue='Heart_Risk', palette='Set2', legend=False)
plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
plt.savefig('static/target_distribution.png')
plt.close()

# 3. Pairplot (only with selected important features for faster plotting)
sns.pairplot(data[['Age', 'Chronic_Stress', 'Fatigue', 'Heart_Risk']], hue='Heart_Risk', palette='Set2', height=2)
plt.savefig('static/pairplot.png')
plt.close()

# 4. Distribution of numerical features (using a small subset of features)
numerical_features = ['Age', 'Chronic_Stress', 'Fatigue', 'Shortness_of_Breath']
for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.histplot(data[feature], kde=True, color='blue', bins=20)
    plt.title(f'Distribution of {feature}')
    plt.savefig(f'static/{feature}_distribution.png')
    plt.close()

# 5. Box plots for numerical features
for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Heart_Risk', y=feature, data=data, palette='Set2')
    plt.title(f'{feature} Distribution by Heart Risk')
    plt.savefig(f'static/{feature}_boxplot.png')
    plt.close()

# 6. Bar plots for categorical features
categorical_features = ['Gender', 'Smoking', 'Family_History']
for feature in categorical_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=feature, hue='Heart_Risk', data=data, palette='Set2', legend=False)
    plt.title(f'{feature} Count by Heart Risk')
    plt.savefig(f'static/{feature}_countplot.png')
    plt.close()

# 7. Pairplot for selected features (Age, Chronic_Stress, Fatigue, etc.)
sns.pairplot(data[['Age', 'Chronic_Stress', 'Fatigue', 'Heart_Risk']], hue='Heart_Risk', palette='Set2', height=2)
plt.savefig('static/numerical_pairplot.png')
plt.close()
