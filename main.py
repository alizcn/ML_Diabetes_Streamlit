import numpy as np
import pickle
import streamlit as st

machine_learning_model = pickle.load(open('knnmodel.pickle', 'rb'))
scaler = pickle.load(open('sc.pickle', 'rb'))


def main():
    st.title("Diabetes Prediction")
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age Of The Person")

    diagnosis = ""

    if st.button("Diabetes Test Result"):
        diagnosis = machine_learning_model.predict(scaler.transform(
            np.array(
                [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])))

    if (diagnosis == 0):
        st.success('The person is not diabetic')
    elif (diagnosis == 1):
        st.success('The person is diabetic')
    else:
        st.success("Check The Values")


if __name__ == "__main__":
    main()
