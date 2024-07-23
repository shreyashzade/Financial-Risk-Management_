# model_performance.py

import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def app(model, X_test, y_test):
    st.title('Model Performance Metrics')

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