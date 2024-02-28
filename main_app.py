import streamlit as st
from streamlit_option_menu import option_menu
from home import home_page
from login_signup import login_signup_page
from Patient_info import patient_info
from data_analysis import visual_analysis_page
from Breast_Cancer import prediction
from Breast_Cancer_Multi_Model import multiple_model_page
from about import about_page
selected = option_menu(
                menu_title=None,  # required
                options=["Home", "Login/Signup", "Patient Info", "Visual Analysis", "Prediction", "Multi-Model Analysis", "About"],  # required
                icons=["house", "sign-turn-right-fill", "info-circle-fill", "clipboard-data-fill", "search-heart-fill", "code-square", "file-person"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
            )

if selected == "Home":
    home_page() 
    st.subheader("Page No: 1")
if selected == "Login/Signup":
    login_signup_page()
    st.subheader("Page No: 2")
if selected == "Patient Info":
    patient_info()
    st.subheader("Page No: 3")
if selected == "Visual Analysis":
    visual_analysis_page()
    st.subheader("Page No: 4")
if selected == "Prediction":
    prediction()
    st.subheader("Page No: 5")
if selected == "Multi-Model Analysis":
    multiple_model_page()
    st.subheader("Page No: 6")
if selected == "About":
    about_page()
    st.subheader("Page No: 7")
    
