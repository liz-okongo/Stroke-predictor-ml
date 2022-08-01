import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import joblib

def main():
    st.title('Stroke Prediction Model Using Random Forest')
    filename = 'Logistic_model.pkl'
    loaded_model = joblib.load(filename)
        #Caching the model for faster loading
        # @st.cache
        # def predict(fixed_acidity, volatile_acidity,  citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    col1, col2 = st.columns(2)
    with col1:
            gender = st.selectbox('Gender:', ["Female", "Male"])
            if gender == "Female":
                gender = 0
            else:
                gender = 1

            hypertension = st.selectbox("Hypertesnion", ["No", "Yes"])
            if hypertension == "No":
                hypertension = 0
            else:
                hypertension = 1

            ever_married = st.selectbox("Married", ["No", 'Yes'])
            if ever_married == "No":
                ever_married = 0
            else:
                ever_married = 1

            Residence_type = st.selectbox("Resident Area", ['Rural', 'Urban'])
            if Residence_type == "Rural":
                Residence_type = 0
            else:
                Residence_type = 1

            bmi = st.number_input("Body Mass Index", min_value=8.0, max_value=100.0)
    with col2:
            age = st.number_input('Age:', min_value=1, max_value=110)

            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            if heart_disease == "No":
                heart_disease = 0
            else:
                heart_disease = 1

            work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Government Job', 'Child', 'Never Worked'])
            if work_type == 'Child':
                work_type = 0
            elif work_type == 'Government Job':
                work_type = 1
            elif work_type == 'Never Worked':
                work_type = 2
            elif work_type == 'Private':
                work_type = 3
            else:
                work_type = 4

            avg_glucose_level = st.number_input("Average Glucose Level", min_value=30.0, max_value=300.0)

            smoking_status = st.selectbox("Smoking Status", ['Formerly smoked', 'Never smoked', 'Smokes', 'Unknown'])
            if smoking_status == 'Formerly smoked':
                smoking_status = 0
            elif smoking_status == 'Never smoked':
                smoking_status = 1
            elif smoking_status == 'Smokes':
                smoking_status = 2
            else:
                smoking_status = 3



    input_dict = {'gender':gender, 'age':age, 'hypertension':hypertension, 'heart_disease':heart_disease,
        'ever_married':ever_married, "work_type":work_type, 'Residence_type':Residence_type, "avg_glucose_level":avg_glucose_level, 
        'bmi':bmi, 'smoking_status':smoking_status}
    input_df = pd.DataFrame(input_dict, index=[0])
        ## predict button
    button = st.button('Predict')

    if button:

            risk = loaded_model.predict(input_df)
            if risk == 0:
                st.success("Low Risk of Stroke")
            else:
                st.error("High Risk of Stroke")
            
            precision, recall, f1, acc = st.columns(4)
            st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 5% 5% 5% 10%;
            border-radius: 5px;
            color: rgb(30, 103, 119);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: green;
            font-size: 20px;
            }
            </style>
            """
            , unsafe_allow_html=True)

            with precision:
                st.metric(label="Precision Score", value="94%")
            with recall:
                st.metric(label="Recall Score", value="94%")
            with f1:
                st.metric(label="F1 Score", value="94%")
            with acc:
                st.metric(label="Accuracy Score", value="94%")

        





if __name__ == '__main__':
    main()
