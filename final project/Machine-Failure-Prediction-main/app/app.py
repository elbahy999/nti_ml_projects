import streamlit as st
import joblib
import numpy as np

# Load ONLY model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Machine Failure Predictor", page_icon="🛠️", layout="wide")

st.sidebar.title("Model Info")
st.sidebar.metric("Model Type", "Random Forest")
st.sidebar.metric("Test Accuracy", "98.1%")
st.sidebar.metric("Precision", "98.0%")
st.sidebar.metric("Recall", "98.1%")

st.title("🛠️ Machine Failure Predictor")
st.write("Enter the sensor data below to predict if a machine failure is likely.")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("Air temperature [K]", value=300.0, min_value=290.0, max_value=310.0)
    process_temp = st.number_input("Process temperature [K]", value=310.0, min_value=300.0, max_value=320.0)
    rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0, min_value=500.0, max_value=3000.0)

with col2:
    torque = st.number_input("Torque [Nm]", value=40.0, min_value=0.0, max_value=100.0)
    tool_wear = st.number_input("Tool wear [min]", value=0.0, min_value=0.0, max_value=300.0)

if st.button("Predict Failure", type="primary"):
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.write("---")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **Warning: Machine Failure Predicted!**")
        st.metric("Failure Probability", f"{prediction_proba[0][1]:.1%}")
    else:
        st.success(f"✅ **Safe: No Failure Predicted**")
        st.metric("Safety Confidence", f"{prediction_proba[0][0]:.1%}")

st.info("This model uses a Random Forest Classifier trained on the AI4I 2020 Predictive Maintenance dataset.")
