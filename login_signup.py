import streamlit as st

def login_signup_page():
    st.title('Welcome to Cancer Detection App')

    st.subheader('Login/Signup to continue')
        
    choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
    if choice =='Login':
        username = st.text_input('Username')
        password = st.text_input('Password', type = 'password')
        if st.button('Login'):
            st.success(f"{username} you have successfully logged in....")

    else:
        username = st.text_input('Username')
        password = st.text_input('Password', type = 'password')
        if st.button('Create my account'):
            st.success(f"{username} you have successfully Signed up and Logged in....")

login_signup_page()
