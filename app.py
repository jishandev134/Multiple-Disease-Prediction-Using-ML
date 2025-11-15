import os
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Prediction of Disease Outbreaks', layout='wide', page_icon='🧑‍⚕️')

# --- SET UP FILE PATHS ---
model_dir = "Saved Models"
diabetes_model_path = os.path.join(model_dir, "diabetes_model.sav")
diabetes_scaler_path = os.path.join(model_dir, "diabetes_scaler.sav")
heart_model_path = os.path.join(model_dir, "heart_model.sav")
heart_scaler_path = os.path.join(model_dir, "heart_scaler.sav")

# --- LOAD MODELS AND SCALERS ---
try:
    diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
    diabetes_scaler = pickle.load(open(diabetes_scaler_path, 'rb'))
    heart_model = pickle.load(open(heart_model_path, 'rb'))
    heart_scaler = pickle.load(open(heart_scaler_path, 'rb'))
    
except FileNotFoundError as e:
    st.error(f"Error: A model or scaler file was not found.")
    st.error(f"Details: {e}")
    st.info(f"Please make sure the following files are in the '{model_dir}' folder:")
    st.info(f"  - diabetes_model.sav")
    st.info(f"  - diabetes_scaler.sav")
    st.info(f"  - heart_model.sav")
    st.info(f"  - heart_scaler.sav")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()


# --- SIDEBAR FOR NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart'],
        default_index=0
    )

# --- DIABETES PREDICTION PAGE ---
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=2, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, value=120)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, value=70)
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, value=20)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, value=79)
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, value=25.0, format="%.1f")
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, value=0.5, format="%.2f")
    with col2:
        Age = st.number_input('Age of the Person', min_value=1, max_value=120, value=33, step=1)

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                      BMI, DiabetesPedigreeFunction, Age)
        input_data_as_numpy_array = np.asarray(user_input)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        input_data_scaled = diabetes_scaler.transform(input_data_reshaped)
        diab_prediction = diabetes_model.predict(input_data_scaled)
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
    st.success(diab_diagnosis)

# --- HEART DISEASE PREDICTION PAGE ---
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=55, step=1)
    with col2:
        sex = st.selectbox('Sex', (0, 1), format_func=lambda x: 'Female' if x == 0 else 'Male', index=1)
    with col3:
        cp = st.number_input('Chest Pain types (cp)', min_value=0, max_value=3, value=0, step=1)
    with col1:
        trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=130)
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=250)
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: 'False' if x == 0 else 'True', index=0)
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results (restecg)', min_value=0, max_value=2, value=1, step=1)
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved (thalach)', min_value=60, max_value=220, value=150)
    with col3:
        exang = st.selectbox('Exercise Induced Angina (exang)', (0, 1), format_func=lambda x: 'No' if x == 0 else 'Yes', index=0)
    with col1:
        oldpeak = st.number_input('ST depression (oldpeak)', min_value=0.0, max_value=7.0, value=1.0, format="%.1f")
    with col2:
        slope = st.number_input('Slope of peak exercise ST segment (slope)', min_value=0, max_value=2, value=2, step=1)
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy (ca)', min_value=0, max_value=4, value=0, step=1)
    with col1:
        thal = st.number_input('Thal (0=normal, 1=fixed, 2=reversable)', min_value=0, max_value=3, value=2, step=1)

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = (age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal)
        input_data_as_numpy_array = np.asarray(user_input)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        input_data_scaled = heart_scaler.transform(input_data_reshaped)
        heart_prediction = heart_model.predict(input_data_scaled)
        heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
    st.success(heart_diagnosis)
