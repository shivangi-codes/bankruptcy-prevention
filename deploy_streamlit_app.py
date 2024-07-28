import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the models
models = {
    'KNN': pickle.load(open('knn.pkl', 'rb')),
    'Logistic Regression': pickle.load(open('logistic_regression.pkl', 'rb')),
    'Decision Tree': pickle.load(open('decision_tree.pkl', 'rb')),
    'Random Forest': pickle.load(open('random_forest.pkl', 'rb')),
    'SVM': pickle.load(open('svm.pkl', 'rb'))
}

# Load the scaler
scaler = StandardScaler()

def main():
    st.title("Bankruptcy Prediction App")

    # User input features
    st.sidebar.header("User Input Features")
    
    financial_flexibility = st.sidebar.slider('Financial Flexibility', 0, 10, 5)
    industrial_risk = st.sidebar.slider('Industrial Risk', 0, 10, 5)
    credibility = st.sidebar.slider('Credibility', 0, 10, 5)
    operating_risk = st.sidebar.slider('Operating Risk', 0, 10, 5)

    features = pd.DataFrame({
        'financial_flexibility': [financial_flexibility],
        'industrial_risk': [industrial_risk],
        'credibility': [credibility],
        'operating_risk': [operating_risk]
    })

    # Preprocess the features
    features_scaled = scaler.fit_transform(features)

    # Model selection
    model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
    model = models[model_choice]

    if st.button('Predict'):
        prediction = model.predict(features_scaled)
        result = "Bankrupt" if prediction[0] == 0 else "Not Bankrupt"
        st.write(f"The model predicts: **{result}**")

if __name__ == '__main__':
    main()

    