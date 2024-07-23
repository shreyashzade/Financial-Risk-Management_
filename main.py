# main.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Set Streamlit page configuration
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

# Main app
st.title('Financial Risk Assessment')
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Assessment', 'Results', 'Model Performance'])

if page == 'Home':
    import home
    home.app()
elif page == 'Assessment':
    import assessment
    assessment.app(model, df)
elif page == 'Results':
    import results
    results.app()
elif page == 'Model Performance':
    import model_performance
    model_performance.app(model, X_test, y_test)