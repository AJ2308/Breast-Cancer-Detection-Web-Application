import streamlit as st
from PIL import Image

def about_page():
    st.title("About Our Breast Cancer Detection Project")

    st.write("Welcome to our innovative machine learning project aimed at enhancing breast cancer detection through advanced data analysis and predictive modeling. With a user-friendly interface and powerful algorithms, our platform is designed to assist healthcare professionals and individuals alike in early detection and accurate diagnosis.")

    st.subheader("Homepage :")
    st.write("Our homepage serves as the gateway to our platform, offering an introduction to our project's objectives, features, and benefits. Users can navigate through various sections to learn more about our methodology, team, and how to get started.")

    st.subheader("Login/Signup Section :")
    st.write("For a personalized experience and access to exclusive features, users can create an account or log in securely. This section ensures data privacy and enables users to track their activities, save preferences, and receive updates.")

    st.subheader("Visual Analysis :")
    st.write("Explore an interactive visual analysis of breast cancer detection features, providing insights into crucial indicators and patterns. Through intuitive charts, graphs, and diagrams, users can gain a deeper understanding of the data driving our predictive models.")

    st.subheader("Prediction :")
    st.write("Enter patient data and receive instant predictions regarding the likelihood of breast cancer. Our advanced algorithms analyze input variables such as age, family history, and genetic markers to generate accurate assessments, aiding in early intervention and treatment planning.")

    st.subheader("Multiple Model Prediction :")
    st.write("Experience the power of multiple model prediction as our platform leverages diverse machine learning algorithms to enhance accuracy and reliability. Compare results from various models to make informed decisions and improve diagnostic outcomes.")

    st.subheader("Inspired by Data Professor:")
    st.write("Our project draws inspiration from Data Professor, a leading figure in the field of data science education and innovation. His dedication to making complex concepts accessible and his commitment to empowering learners have inspired us to pursue excellence in our breast cancer detection project. For more insightful content on data science and machine learning, visit Data Professor's YouTube channel. Join us in the fight against breast cancer by harnessing the potential of machine learning and data-driven insights. Together, we can make a difference in early detection and improve patient outcomes.")
    st.write("Chanin Nantasenamat, Ph.D. is a Developer Advocate, YouTuber (Data Professor channel) and ex-Professor of Bioinformatics. He creates educational video (>300 videos) and written (~80 blogs) contents about data science, machine learning and bioinformatics on YouTube and various blogging platforms. He's also published >160 research/review articles in peer-review journals at the interface of machine learning, biology and chemistry for exploring the underlying origins of protein function where such understanding holds great implication for drug discovery.")
    st.write("Youtube Channel Link: `http://bit.ly/dataprofessor`")

    st.subheader("About the Project Maker :")
    st.write("- `Name:` Aditya Jaiswal")
    st.write("- `Qualification:` MBA in Business Analytics & Data Science")
    self_image = Image.open("AJ_professional.jpg")
    scaled_image = self_image.resize((295,291))
    st.image(scaled_image, caption='Business Analytics & Data Science', use_column_width=True)
    st.write("Aditya's expertise in data science and business analytics drives the development of innovative solutions for healthcare challenges, such as early breast cancer detection.")
about_page()


