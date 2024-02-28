import streamlit as st


def patient_info():
    st.title('Patient Details')

    st.subheader('Enter Patient Details')

    # Creating our form fields
    with st.form('Details', clear_on_submit = True):
        first_name = st.text_input('First Name')
        last_name = st.text_input('Last Name')
        gender = st.selectbox('Gender', ['Male', 'Female', 'Others'])
        age = st.slider('Age', min_value = 10, max_value = 100, step = 1)
        
        if st.form_submit_button('Submit'):
            st.write(f'Name: {first_name} {last_name}')
            st.write(f'Gender: {gender}')
            st.write(f'Age: {age}')

            st.success("Patient Details entered successfully...")

patient_info()
