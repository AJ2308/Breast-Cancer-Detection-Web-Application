import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

df = pd.read_csv("C:\\Users\\ADITYA\\Downloads\\archive (6)\\data.csv")
df.drop(columns=['id'], inplace=True)

lb = LabelEncoder()
df['diagnosis'] = lb.fit_transform(df['diagnosis'])

x = df.iloc[:, 1: 7].values
y = df.iloc[:, 0].values

sc = StandardScaler()
x[:, 0:4] = sc.fit_transform(x[:, 0:4])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

class_names = ['malignant', 'benign']

if st.sidebar.checkbox("Show X_train/Y_train", False):
            st.subheader('X_train')
            st.dataframe(x_train)
            st.subheader('Y_train')
            st.dataframe(y_train)

# def plot_metrics(metrics_list):
#         if 'Confusion Matrix' in metrics_list:
#             st.subheader("Confusion Matrix")
#             cm = confusion_matrix(y_test, y_pred)
#             fig,ax = plt.subplots()
#             sns.heatmap(cm, annot = True)
#             # plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
#             ax.figure.savefig('file.png')
#             st.pyplot(fig)
#         if 'Precision-Recall Curve' in metrics_list:
#             st.subheader('Precision-Recall Curve')
            
#             precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

#             fig = px.area(
#                 x=recall, y=precision,
#                 title=f'Precision-Recall Curve (AUC={auc(precision, recall):.4f})',
#                 labels=dict(x='Recall', y='Precision'),
#                 width=700, height=500
#                 )
#             fig.add_shape(
#                 type='line', line=dict(dash='dash'),
#                 x0=0, x1=1, y0=1, y1=0
#                 )
#             fig.update_yaxes(scaleanchor="x", scaleratio=1)
#             fig.update_xaxes(constrain='domain')
#             st.write(fig)
            
            
#         if 'ROC Curve' in metrics_list:
#             fpr, tpr, thresholds = roc_curve(y_test, y_pred)

#             fig = px.area(
#                x=fpr, y=tpr,
#                title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
#                labels=dict(x='False Positive Rate', y='True Positive Rate'),
#                width=700, height=500
#                )       
#             fig.add_shape(
#                 type='line', line=dict(dash='dash'),
#                 x0=0, x1=1, y0=0, y1=1
#                 )

#             fig.update_yaxes(scaleanchor="x", scaleratio=1)
#             fig.update_xaxes(constrain='domain')
#             st.write(fig)
        
#         if 'Training and Test accuracies' in metrics_list:
#             mal_train_X = x_train[y_train==0]
#             mal_train_y = y_train[y_train==0]
#             ben_train_X = x_train[y_train==1]
#             ben_train_y = y_train[y_train==1]
            
#             mal_test_X = x_test[y_test==0]
#             mal_test_y = y_test[y_test==0]
#             ben_test_X = x_test[y_test==1]
#             ben_test_y = y_test[y_test==1]
            
#             scores = [model.score(mal_train_X, mal_train_y), model.score(ben_train_X, ben_train_y), model.score(mal_test_X, mal_test_y), model.score(ben_test_X, ben_test_y)]

#             fig,ax = plt.subplots()
        
#     # Plot the scores as a bar chart
#             bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

#     # directly label the score onto the bars
#             for bar in bars:
#                 height = bar.get_height()
#                 plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), ha='center', color='w', fontsize=11)

#     # remove all the ticks (both axes), and tick labels on the Y axis
#             plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

#     # remove the frame of the chart
#             for spine in plt.gca().spines.values():
#                 spine.set_visible(False)

#             plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
#             plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
#             ax.xaxis.set_tick_params(length=0)
#             ax.yaxis.set_tick_params(length=0)
#             ax.figure.savefig('file1.png')
#             st.pyplot(fig)

def multiple_model_page():
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

   temp_x = df.iloc[:, 1: 7]

   user_inputs = pd.DataFrame(data = np.array([Radius_Mean, Texture_Mean, Perimeter_Mean, Area_Mean, Smoothness_Mean, Compactness_Mean]).reshape(1,6), columns=temp_x.columns)

   st.write('Breast Cancer Features')
   st.write(user_inputs)
   
   benign_image = Image.open("C:\\Users\\ADITYA\\Downloads\\Benign_image.jpg")
   benign_image1 = Image.open("C:\\Users\\ADITYA\\Downloads\\Benign_image1.jpg")
   benign_image2 = Image.open("C:\\Users\\ADITYA\\Downloads\\Benign_image.2jpg.jpg")

   malignant_image = Image.open("C:\\Users\\ADITYA\\Downloads\\Malignant_image.jpg")
   malignant_image1 = Image.open("C:\\Users\\ADITYA\\Downloads\\Malignant_image.1jpg.jpg")
   malignant_image2 = Image.open("C:\\Users\\ADITYA\\Downloads\\Malignant_image.2jpg.jpg")


   st.subheader("Choose Classifier")
   classifier = st.selectbox("Classifier", ["Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", 'KNN', 'Decision Tree', 'Gaussian Naive Bayes'])

   if classifier == 'Support Vector Machine (SVM)':
            st.subheader("Model Hyperparameters")
            #choose parameters
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
            kernel = st.radio("Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
                      
            if st.button("Classify", key='classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                result = model.predict(user_input)
                prediction = result
                
                if prediction == 0:
                    y_pred = model.predict(x_test)
                    st.write('Report Result : Benign Tumor')
                    st.write('Description : It is an abnormal but noncancerous collection of cells. Many benign tumors dont require treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Benign Tumor X-Ray Image')
                    st.image([benign_image, benign_image1, benign_image2])
        
                else:
                    y_pred = model.predict(x_test)
                    st.write('Report Result : Malignant Tumor')
                    st.write('Description : It is made of cancer cells that can grow uncontrollably and invade nearby tissues. It requires immediate treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Malignant Tumor X-Ray Image')
                    st.image([malignant_image, malignant_image1, malignant_image2])
                
    
   if classifier == 'Logistic Regression':
            st.subheader("Model Hyperparameters")
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.slider("Maximum number of iterations", 100, 500, key='max_iter')
            
            if st.button("Classify", key='classify'):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                result = model.predict(user_input)
                prediction = result
                if prediction == 0:
                    st.write('Report Result : Benign Tumor')
                    st.write('Description : It is an abnormal but noncancerous collection of cells. Many benign tumors dont require treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Benign Tumor X-Ray Image')
                    st.image([benign_image, benign_image1, benign_image2])
                else:
                    st.write('Report Result : Malignant Tumor')
                    st.write('Description : It is made of cancer cells that can grow uncontrollably and invade nearby tissues. It requires immediate treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Malignant Tumor X-Ray Image')
                    st.image([malignant_image, malignant_image1, malignant_image2])
                
                
   if classifier == 'Random Forest':
            st.subheader("Model Hyperparameters")
            n_estimators = st.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
            bootstrap = st.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
            
            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                result = model.predict(user_input)
                prediction = result
                if prediction == 0:
                    st.write('Report Result : Benign Tumor')
                    st.write('Description : It is an abnormal but noncancerous collection of cells. Many benign tumors dont require treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Benign Tumor X-Ray Image')
                    st.image([benign_image, benign_image1, benign_image2])
                else:
                    st.write('Report Result : Malignant Tumor')
                    st.write('Description : It is made of cancer cells that can grow uncontrollably and invade nearby tissues. It requires immediate treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Malignant Tumor X-Ray Image')
                    st.image([malignant_image, malignant_image1, malignant_image2])
                
   if classifier == 'KNN':
            st.subheader("Model Hyperparameters")
            n_neighbors = st.number_input("Number of neighbors", 1, 100, step=1, key='n_neighbors')
            
            if st.button("Classify", key='classify'):
                st.subheader("KNN Results")
                model = KNeighborsClassifier(n_neighbors = n_neighbors )
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                result = model.predict(user_input)
                prediction = result
                if prediction == 0:
                    st.write('Report Result : Benign Tumor')
                    st.write('Description : It is an abnormal but noncancerous collection of cells. Many benign tumors dont require treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Benign Tumor X-Ray Image')
                    st.image([benign_image, benign_image1, benign_image2])

                else:
                    st.write('Report Result : Malignant Tumor')
                    st.write('Description : It is made of cancer cells that can grow uncontrollably and invade nearby tissues. It requires immediate treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Malignant Tumor X-Ray Image')
                    st.image([malignant_image, malignant_image1, malignant_image2])
                
   if classifier == 'Decision Tree':
            st.subheader("Model Hyperparameters")
            
            max_depth = st.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
            criterion = st.radio("Criterion", ("gini", "entropy"), key='criterion')
            splitter = st.radio("Splitter", ("best", "random"), key='splitter')
               
            if st.button("Classify", key='classify'):
                st.subheader("Decision Tree Results")
                model = DecisionTreeClassifier(max_depth= max_depth, criterion= criterion, splitter= splitter )
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                result = model.predict(user_input)
                prediction = result
                if prediction == 0:
                    st.write('Report Result : Benign Tumor')
                    st.write('Description : It is an abnormal but noncancerous collection of cells. Many benign tumors dont require treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Benign Tumor X-Ray Image')
                    st.image([benign_image, benign_image1, benign_image2])

                else:
                    st.write('Report Result : Malignant Tumor')
                    st.write('Description : It is made of cancer cells that can grow uncontrollably and invade nearby tissues. It requires immediate treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Malignant Tumor X-Ray Image')
                    st.image([malignant_image, malignant_image1, malignant_image2])
                
   if classifier == 'Gaussian Naive Bayes':
            st.subheader("Model Hyperparameters")
            
            if st.button("Classify", key='classify'):
                st.subheader("Gaussian Naive Bayes Results")
                model = GaussianNB()
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                result = model.predict(user_input)
                prediction = result
                if prediction == 0:
                    st.write('Report Result : Benign Tumor')
                    st.write('Description : It is an abnormal but noncancerous collection of cells. Many benign tumors dont require treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Benign Tumor X-Ray Image')
                    st.image([benign_image, benign_image1, benign_image2])
                else:
                    st.write('Report Result : Malignant Tumor')
                    st.write('Description : It is made of cancer cells that can grow uncontrollably and invade nearby tissues. It requires immediate treatment.')
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    st.header('Malignant Tumor X-Ray Image')
                    st.image([malignant_image, malignant_image1, malignant_image2])

multiple_model_page()
