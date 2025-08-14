
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Prediction of Disease Outbreaks', layout='wide', page_icon='🧑‍⚕️')

# Importing saved ML models
diabetes_model = pickle.load(open(f"Saved Models/diabetes_model.sav","rb"))
heart_model = pickle.load(open(f"Saved Models/heart_model.sav","rb"))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', value='2')
    with col2:
        Glucose = st.text_input('Glucose Level', value='120')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', value='70')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', value='20')
    with col2:
        Insulin = st.text_input('Insulin Level', value='79')
    with col3:
        BMI = st.text_input('BMI value', value='25.0')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value='0.5')
    with col2:
        Age = st.text_input('Age of the Person', value='33')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age', value='55')
    with col2:
        sex = st.text_input('Sex (1=male, 0=female)', value='1')
    with col3:
        cp = st.text_input('Chest Pain types', value='0')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', value='130')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', value='250')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', value='0')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results', value='1')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved', value='150')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=Yes,0=No)', value='0')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise', value='1.0')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment', value='2')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy', value='0')
    with col1:
        thal = st.text_input('thal: 0=normal;1=fixed defect;2=reversable defect', value='2')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])
        heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'

    st.success(heart_diagnosis)
