# assessment.py

import streamlit as st
import pandas as pd

def app(model, df):
    st.title('Financial Risk Assessment')
    st.header('Enter Customer Information')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100)
        income = st.number_input('Income', min_value=0)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
        loan_amount = st.number_input('Loan Amount', min_value=0)
        years_at_job = st.number_input('Years at Current Job', min_value=0)
        dti_ratio = st.number_input('Debt-to-Income Ratio', min_value=0.0, max_value=1.0)

    with col2:
        assets_value = st.number_input('Assets Value', min_value=0)
        dependents = st.number_input('Number of Dependents', min_value=0)
        previous_defaults = st.number_input('Previous Defaults', min_value=0)
        marital_status_change = st.number_input('Marital Status Change', min_value=0, max_value=1)
        gender = st.selectbox('Gender', df['Gender'].unique())
        education = st.selectbox('Education Level', df['Education Level'].unique())

    with col3:
        marital_status = st.selectbox('Marital Status', df['Marital Status'].unique())
        loan_purpose = st.selectbox('Loan Purpose', df['Loan Purpose'].unique())
        employment_status = st.selectbox('Employment Status', df['Employment Status'].unique())
        payment_history = st.selectbox('Payment History', df['Payment History'].unique())
        city = st.selectbox('City', df['City'].unique())
        state = st.selectbox('State', df['State'].unique())
        country = st.selectbox('Country', df['Country'].unique())

    if st.button('Predict Risk Rating'):
        input_data = pd.DataFrame({
            'Age': [age], 'Income': [income], 'Credit Score': [credit_score], 'Loan Amount': [loan_amount],
            'Years at Current Job': [years_at_job], 'Debt-to-Income Ratio': [dti_ratio], 'Assets Value': [assets_value],
            'Number of Dependents': [dependents], 'Previous Defaults': [previous_defaults],
            'Marital Status Change': [marital_status_change], 'Gender': [gender], 'Education Level': [education],
            'Marital Status': [marital_status], 'Loan Purpose': [loan_purpose], 'Employment Status': [employment_status],
            'Payment History': [payment_history], 'City': [city], 'State': [state], 'Country': [country]
        })

        prediction = model.predict(input_data)
        st.session_state.prediction = prediction[0]
        st.success(f"Prediction complete. Please go to the Results page to view the prediction.")