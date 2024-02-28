import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data.csv")

df.drop(columns=['id'], inplace=True)

def prediction():
   lb = LabelEncoder()
   df['diagnosis'] = lb.fit_transform(df['diagnosis'])

   x = df.iloc[:, 1: 7].values
   y = df.iloc[:, 0].values


   sc = StandardScaler()
   x[:, 0:4] = sc.fit_transform(x[:, 0:4])

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)



   rfc = RandomForestClassifier()

   rfc.fit(x_train, y_train)

   st.title('Breast Cancer Detection')

   Radius_Mean = st.slider("Radius Mean", min_value = 6.00, max_value = 29.00, step = 0.50)
   Texture_Mean = st.slider("Texture Mean", min_value = 9.00, max_value = 40.00, step = 0.50)
   Perimeter_Mean = st.slider("Perimeter Mean", min_value = 43.00, max_value = 189.00, step = 0.50)
   Area_Mean = st.slider("Area Mean", min_value = 143.00, max_value = 2501.00, step = 0.50)
   Smoothness_Mean = st.slider('Smoothness Mean', min_value = 0.01, max_value = 0.18, step = 0.01)
   Compactness_Mean = st.slider('Compactness Mean', min_value = 0.005, max_value = 0.400, step = 0.05)

   scaling_user_input = [[Radius_Mean, Texture_Mean, Perimeter_Mean, Area_Mean]]
   scaled_user_input = sc.transform(scaling_user_input)

   non_scaled_input = np.array([[Smoothness_Mean, Compactness_Mean]])

   user_input = np.concatenate((scaled_user_input, non_scaled_input), axis=1)

   loaded_model = pickle.load(open('rfc_model.pickle', 'rb'))
   result = loaded_model.predict(user_input)
   accuracy = loaded_model.score(x_test, y_test)

   temp_x = df.iloc[:, 1: 7]

   user_inputs = pd.DataFrame(data = np.array([Radius_Mean, Texture_Mean, Perimeter_Mean, Area_Mean, Smoothness_Mean, Compactness_Mean]).reshape(1,6), columns=temp_x.columns)

   st.write('Breast Cancer Features')
   st.write(user_inputs)

   benign_image = Image.open("Benign_image.jpg")
   benign_image1 = Image.open("Benign_image1.jpg")
   benign_image2 = Image.open("Benign_image.2jpg.jpg")

   malignant_image = Image.open("Malignant_image.jpg")
   malignant_image1 = Image.open("Malignant_image.1jpg.jpg")
   malignant_image2 = Image.open("Malignant_image.2jpg.jpg")

   st.header('Prediction')
   st.write('Prediction of diagnosis based on 4 key criteria as determind by data analysis of the Wisconsin Breast Cancer Dataset. Change the Input Parameters in the sidebar to see the diagnosis change.')
   st.write('**0 = Benign, 1 = Malignant**')

   if st.button("Predict"):
      accuracy_percentage = (f' The Accuracy of the Model : {accuracy * 100}%')
      prediction = result
      if prediction == 0:
         st.write('Report Result : Benign Tumor')
         st.write('Description : It is an abnormal but noncancerous collection of cells. Many benign tumors dont require treatment.')
         st.write(f'{accuracy_percentage}')
         st.header('Benign Tumor X-Ray Image')
         st.image([benign_image, benign_image1, benign_image2])
      else:
         st.write('Report Result : Malignant Tumor')
         st.write('Description : It is made of cancer cells that can grow uncontrollably and invade nearby tissues. It requires immediate treatment.')
         st.write(f'{accuracy_percentage}')
         st.header('Malignant Tumor X-Ray Image')
         st.image([malignant_image, malignant_image1, malignant_image2])
   

prediction()


