import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:\\Users\\ADITYA\\Downloads\\archive (6)\\data.csv")
df.drop(columns=['id'], inplace=True)

lb = LabelEncoder()
df['diagnosis'] = lb.fit_transform(df['diagnosis'])


def visual_analysis_page():
    if st.sidebar.checkbox("Show Raw Data", False):
            st.subheader('Breast Cancer Dataset')
            st.dataframe(df)
            
    if st.sidebar.checkbox("Show Features", False):
            cancer = df
            st.subheader('Features')
            features = pd.DataFrame(cancer.iloc[:, 1:7])
            st.dataframe(features)
            
    
    plots =  st.sidebar.multiselect("Plots", ('Scatter Matrix', 'Number of Malignant and Benign','Heatmap','Mean radius vs Mean area','Mean smoothness vs Mean area'))
        
        
    if st.sidebar.button("Plot", key='plotss'):
            with st.spinner('Wait for it...'):
                time.sleep(5)
 
            if 'Number of Malignant and Benign' in plots:
                st.subheader("Malignant and Benign Count")
                fig,ax = plt.subplots()
                
                ma = len(df[df['diagnosis']==1])
                be = len(df[df['diagnosis']==0])
                count=[ma,be]
                bars = plt.bar(np.arange(2), count, color=['#000099','#ffff00'])
                ##show value in bars
                for bar in bars:
                    height = bar.get_height()
                    plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), ha='center', color='black', fontsize=11)
                plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                plt.xticks(ticks=[0,1])
                ax.set_ylabel('Count')
                ax.set_xlabel('Target')
                ##remove dashes from frame
                ax.xaxis.set_tick_params(length=0)
                ax.yaxis.set_tick_params(length=0)
                st.pyplot(fig)
                
            
            if 'Scatter Matrix' in plots:
                st.subheader("Scatter Matrix")
                fig = px.scatter_matrix(df,dimensions=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean'],color="diagnosis",width = 800,height = 700)
                st.write(fig)
            
            if 'Heatmap' in plots:
                st.subheader("Heatmap")
                fig=plt.figure(figsize = (30,20))
                hmap=sns.heatmap(df.drop(columns=['diagnosis']).corr(), annot = True,cmap= 'Blues',annot_kws={"size": 18})
                hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize = 25)
                hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize = 25)
                st.pyplot(fig)
            if 'Mean radius vs Mean area' in plots:
                st.subheader('Cancer Radius and Area')
                fig = plt.figure()
                sns.scatterplot(x=df['radius_mean'],y = df['area_mean'],hue = df['diagnosis'],palette=['#000099','#ffff00'])
                st.pyplot(fig)
            if 'Mean smoothness vs Mean area' in plots:
                st.subheader('Cancer Smoothness and Area')
                fig = plt.figure()
                sns.scatterplot(x=df['smoothness_mean'],y = df['area_mean'],hue = df['diagnosis'],palette=['#000099','#ffff00'])
                st.pyplot(fig)

visual_analysis_page()
