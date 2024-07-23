# results.py

import streamlit as st

def app():
    st.title('Risk Rating Prediction')
    if 'prediction' in st.session_state:
        st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>Predicted Risk Rating: {st.session_state.prediction}</h1>", unsafe_allow_html=True)
    else:
        st.warning("No prediction available. Please go to the Assessment page to make a prediction.")