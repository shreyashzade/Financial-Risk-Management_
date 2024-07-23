# home.py

import streamlit as st

def app():
    st.title('Financial Risk Assessment')
    st.write("Welcome to the Financial Risk Assessment tool. This application helps predict the risk rating of a customer based on various financial and personal factors.")

    # Create two columns: one for the text and one for the image
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Input Terms Explanation")
        st.write("""
        - Age: Customer's age in years
        - Income: Annual income of the customer
        - Credit Score: Customer's credit score (300-850)
        - Loan Amount: Amount of loan requested
        - Years at Current Job: Number of years at the current job
        - Debt-to-Income Ratio: Ratio of monthly debt payments to monthly income
        - Assets Value: Total value of customer's assets
        - Number of Dependents: Number of people depending on the customer's income
        - Previous Defaults: Number of previous loan defaults
        - Marital Status Change: Recent change in marital status (0 for no, 1 for yes)
        - Gender: Customer's gender
        - Education Level: Highest level of education completed
        - Marital Status: Current marital status
        - Loan Purpose: Purpose of the loan
        - Employment Status: Current employment status
        - Payment History: Past payment behavior
        - City, State, Country: Location information
        """)

    with col2:
        st.image("https://static.javatpoint.com/commerce/images/financial-risk-management1.jpg", caption="Placeholder for uploaded image")