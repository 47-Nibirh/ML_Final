import pandas as pd
import numpy as np
import gradio as gr
import pickle

# Load the trained model and feature names
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def predict_charges(age, sex, bmi, children, smoker, region):
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Preprocess similar to training data
    input_data['sex'] = input_data['sex'].map({'female': 0, 'male': 1})
    input_data['smoker'] = input_data['smoker'].map({'no': 0, 'yes': 1})
    input_data = pd.get_dummies(input_data, columns=['region'], drop_first=True)
    
    # BMI category
    input_data['bmi_category'] = pd.cut(input_data['bmi'], bins=[0, 18.5, 25, 30, np.inf], labels=['underweight', 'normal', 'overweight', 'obese'])
    input_data = pd.get_dummies(input_data, columns=['bmi_category'], drop_first=True)
    
    # Ensure all columns match training data
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_names]
    
    # Predict using the best model
    prediction = best_model.predict(input_data)
    return f"Predicted Insurance Charges: ${prediction[0]:.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_charges,
    inputs=[
        gr.Number(label="Age", minimum=18, maximum=100),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI", minimum=10, maximum=50),
        gr.Number(label="Number of Children", minimum=0, maximum=10),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Insurance Charges Predictor",
    description="Predict medical insurance charges based on personal information."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()