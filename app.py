import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model_data():
    # Load the serialized model, scaler and metadata
    return joblib.load('insurance_model.joblib')

# Try to load the model
try:
    assets = load_model_data()
    model = assets['model']
    scaler = assets['scaler']
    features = assets['features']
    num_cols = assets['numerical_cols']
except Exception as e:
    st.error(f"Could not load model file: {e}. Please run the saving cell in the notebook first.")
    st.stop()

st.title("Medical Cost Prediction App")
st.write("Real-time prediction using Gradient Boosting Regressor.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    steps = st.number_input("Daily Steps", 0, 30000, 5000)
    sleep = st.slider("Sleep Hours", 1.0, 12.0, 7.0)
    stress = st.slider("Stress Level", 1, 10, 5)
    visits = st.number_input("Doctor Visits/Year", 0, 20, 2)
    admissions = st.number_input("Hospital Admissions", 0, 10, 0)
    meds = st.number_input("Medication Count", 0, 20, 1)

with col2:
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    asthma = st.selectbox("Asthma", ["No", "Yes"])
    pal = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
    coverage = st.slider("Insurance Coverage %", 0, 100, 80)
    prev_cost = st.number_input("Previous Year Cost", 0.0, 100000.0, 2000.0)
    gender = st.selectbox("Gender", ["Female", "Male"])
    city = st.selectbox("City Type", ["Rural", "Semi-Urban", "Urban"])
    insurance = st.selectbox("Insurance Type", ["Government", "Private"])

if st.button("Predict Medical Cost"):
    pal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Prepare the input dictionary
    input_dict = {
        'age': age, 'bmi': bmi, 'diabetes': 1 if diabetes == 'Yes' else 0, 
        'hypertension': 1 if hypertension == 'Yes' else 0, 
        'heart_disease': 1 if heart_disease == 'Yes' else 0, 
        'asthma': 1 if asthma == 'Yes' else 0, 
        'physical_activity_level': pal_map[pal], 'daily_steps': steps,
        'sleep_hours': sleep, 'stress_level': stress, 'doctor_visits_per_year': visits, 
        'hospital_admissions': admissions, 'medication_count': meds, 
        'insurance_coverage_pct': coverage, 'previous_year_cost': prev_cost,
        'city_type_Semi-Urban': 1 if city == 'Semi-Urban' else 0,
        'city_type_Urban': 1 if city == 'Urban' else 0,
        'insurance_type_Private': 1 if insurance == 'Private' else 0,
        'smoker_Yes': 1 if smoker == 'Yes' else 0,
        'gender_Male': 1 if gender == 'Male' else 0
    }
    
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])[features]
    
    # Scale numeric columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"### Predicted Annual Medical Cost: ${prediction:,.2f}")
