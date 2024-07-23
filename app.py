import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set Streamlit page configuration at the top
st.set_page_config(page_title="Financial Risk Assessment", layout="wide")

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('financial_risk_assessment.csv')

df = load_data()

# Define features and target
X = df.drop('Risk Rating', axis=1)
y = df['Risk Rating']

# Define feature types
numeric_features = ['Age', 'Income', 'Credit Score', 'Loan Amount', 'Years at Current Job',
                    'Debt-to-Income Ratio', 'Assets Value', 'Number of Dependents', 'Previous Defaults', 'Marital Status Change']
categorical_features = ['Gender', 'Education Level', 'Marital Status', 'Loan Purpose', 'Employment Status',
                        'Payment History', 'City', 'State', 'Country']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define and train the model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

# Streamlit app
# Define pages
def home():
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

    if st.button("Proceed to Risk Assessment"):
        st.session_state.page = "assessment"
        st.experimental_rerun()

def assessment():
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
        st.session_state.page = "result"
        st.experimental_rerun()

def result():
    st.title('Risk Rating Prediction')
    st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>Predicted Risk Rating: {st.session_state.prediction}</h1>", unsafe_allow_html=True)
    
    if st.button("Back to Assessment"):
        st.session_state.page = "assessment"
        st.experimental_rerun()

# Navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home()
elif st.session_state.page == "assessment":
    assessment()
elif st.session_state.page == "result":
    result()

# Model performance metrics
if st.session_state.page != "result":
    st.header('Model Performance Metrics')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f'Accuracy: {accuracy:.2f}')
    st.write(f'F1 Score: {f1:.2f}')

    st.subheader('Classification Report')
    st.text(classification_report(y_test, y_pred))

    st.subheader('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(conf_matrix)
