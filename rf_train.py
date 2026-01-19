import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import gradio as gr
import pickle

# Task 1: Data Loading
print("Task 1: Data Loading")
data = pd.read_csv('insurance.csv')
print("First few rows:")
print(data.head())
print(f"Shape: {data.shape}")

# Task 2: Data Preprocessing
print("\nTask 2: Data Preprocessing")

# Reload data for proper preprocessing
data = pd.read_csv('insurance.csv')

# 1. Check for missing values
print("1. Missing values:")
print(data.isnull().sum())

# 2. Encode categorical variables
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# 3. Handle outliers in charges using IQR (done before splitting as it affects target only)
Q1 = data['charges'].quantile(0.25)
Q3 = data['charges'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['charges'] >= lower_bound) & (data['charges'] <= upper_bound)]
print(f"Shape after outlier removal: {data.shape}")

# 4. Feature engineering: BMI categories
data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 25, 30, np.inf], labels=['underweight', 'normal', 'overweight', 'obese'])
data = pd.get_dummies(data, columns=['bmi_category'], drop_first=True)

# Note: Scaling will be done in the pipeline after splitting to avoid data leakage

print("Preprocessing completed. Final shape:", data.shape)

# Task 3: Pipeline Creation
print("\nTask 3: Pipeline Creation")

# Define features and target
X = data.drop('charges', axis=1)
y = data['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with preprocessing
numerical_features = ['age', 'bmi', 'children']
categorical_features = [col for col in X.columns if col not in numerical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', 'passthrough', categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

print("Pipeline created with preprocessing.")

# Task 4: Primary Model Selection
print("\nTask 4: Primary Model Selection")
print("Chosen model: Random Forest Regressor")
print("Justification: Random Forest is suitable for this regression problem because:")
print("- It handles both numerical and categorical features well")
print("- It's robust to outliers and non-linear relationships")
print("- It provides feature importance")
print("- It performs well on tabular data like insurance datasets")
print("- It reduces overfitting compared to single decision trees")

# Task 5: Model Training
print("\nTask 5: Model Training")
pipeline.fit(X_train, y_train)
print("Model trained on training data.")

# Task 6: Cross-Validation
print("\nTask 6: Cross-Validation")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Average CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Task 7: Hyperparameter Tuning
print("\nTask 7: Hyperparameter Tuning")
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Task 8: Best Model Selection
print("\nTask 8: Best Model Selection")
best_model = grid_search.best_estimator_
print("Best model selected based on hyperparameter tuning results.")

# Task 9: Model Performance Evaluation
print("\nTask 9: Model Performance Evaluation")
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test set performance metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)
print("Model saved.")

print("\nAll tasks completed!")

