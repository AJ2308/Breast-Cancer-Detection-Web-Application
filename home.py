import streamlit as st

def home_page():
    st.title("Welcome to BreastCARE: Early Detection Saves Lives")
    st.subheader("Purpose:")
    st.write("BreastCARE is designed to empower individuals by aiding in the early detection of breast cancer through cutting-edge machine learning algorithms. By providing accessible tools for early detection, we aim to improve treatment outcomes and ultimately save lives.")
    
    st.subheader("Why Early Detection Matters:")
    st.write("Early detection of breast cancer significantly enhances treatment options and increases the chances of successful outcomes. By identifying abnormalities at an early stage, individuals can access timely interventions, leading to more effective treatments and improved survival rates. BreastCARE is here to support you in taking proactive steps towards your health and well-being.")
    
    st.subheader("Features of BreastCARE:")
    st.write("- `Streamlined Interface:` User-friendly design for easy navigation.")
    st.write("- `Predictive Algorithms:` State-of-the-art machine learning models for accurate risk assessment.")
    st.write("- `Personalized Recommendations:` Tailored insights and recommendations based on individual risk factors.")
    st.write("- `Educational Resources:` Access to informative content on breast health, screening guidelines, and risk factors.")
    
    st.subheader("Join Us in the Fight Against Breast Cancer:")
    st.write("Together, let’s harness the power of technology to make a difference in the fight against breast cancer. By detecting it early, we can save lives and create a healthier future for all.")
    
    # Embedding a video about cancer awareness
    st.subheader("Cancer Awareness Video")
    st.video("https://www.youtube.com/watch?v=LEpTTolebqo")  # Replace VIDEO_ID with the actual YouTube video ID


home_page()
