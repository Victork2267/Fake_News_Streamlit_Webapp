import pickle
import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

import nltk
import nltk as nlp
import string
import re
import joblib

from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go

deep_trad_model_select = st.sidebar.radio("Use deep learning models or traditional machine learning models for fake news prediction",["Deep learning models","Traditional machine learning models"])

if deep_trad_model_select == "Deep learning models":
    st.header("Fake News Classification app using Deep learning models - Not Ready")

elif deep_trad_model_select == "Traditional machine learning models":
    st.header("Fake News Classification app using traditional machine-learning models")
    prediction_option = st.selectbox("Utilise either 1 model or compare between all models", ["Use 1 model for prediction", "Compare results of all models"])
    if prediction_option == "Use 1 model for prediction":
        model_list = ["Logistic Regression",'Multinomial Naive Bayes Classifier','Gradient Boost Classifier','Decision Tree']
        model_option = st.selectbox("Select the model to use:",model_list)

        if model_option == "Logistic Regression":
            filename = r'models/LR_model.pkl'
            model = pickle.load(open(filename, "rb"))
            st.info("Model {} has been loaded".format(model_option))
        elif model_option == "Multinomial Naive Bayes Classifier":
            filename = r'models/MNVBC_model.pkl'
            model = pickle.load(open(filename, "rb"))
            st.info("Model {} has been loaded".format(model_option))
        elif model_option == "Gradient Boost Classifier":
            filename = r'models/GBC_model.pkl'
            model = pickle.load(open(filename, "rb"))
            st.info("Model {} has been loaded".format(model_option))
        elif model_option == "Decision Tree":
            filename = r'models/DT_model.pkl'
            model = pickle.load(open(filename, "rb"))
            st.info("Model {} has been loaded".format(model_option))

        st.subheader("Input the News content below")

        user_input = st.text_area("Enter your news content here", "Some news",height=200)
        if st.button("predict"):
            prediction = model.predict([user_input])[0]
            prediction_proba = model.predict_proba([user_input])[0]
            class_label = ["fake","true"]
            prob_list = [prediction_proba[0]*100,prediction_proba[1]*100]
            prob_dict = {"true/fake":class_label,"Probability":prob_list}
            df_prob = pd.DataFrame(prob_dict)
            fig = px.bar(df_prob, x='true/fake', y='Probability')
            if prediction_proba[0] > 0.7:
                fig.update_layout(title_text="{} model - prediction probability comparison between true and fake".format(model_option))
                st.info("The {} model predicts that there is a higher {} probability that the news content is fake compared to a {} probability of being true".format(model_option,prediction_proba[0]*100,prediction_proba[1]*100))
            elif prediction_proba[1] > 0.7:
                fig.update_layout(title_text="{} model - prediction probability comparison between true and fake".format(model_option))
                st.info("The {} model predicts that there is a higher {} probability that the news content is true compared to a {} probability of being fake".format(model_option,prediction_proba[1]*100,prediction_proba[0]*100))
            else:
                fig.update_layout(title_text="{} model - prediction probability comparison between true and fake".format(model_option))
                st.info("Your news content is rather abstract, The {} model predicts that there a almost equal {} probability that the news content is true compared to a {} probability of being fake".format(model_option,prediction_proba[1]*100,prediction_proba[0]*100))
            st.plotly_chart(fig, use_container_width=True)
            if prediction == "true":
                st.success('This is not a fake news')
            if prediction == "fake":
                st.warning('This is a fake news')

    elif prediction_option == "Compare results of all models":
        model_list = ["Logistic Regression",'Multinomial Naive Bayes Classifier','Gradient Boost Classifier','Decision Tree']
        model_file_list = [r"models/LR_model.pkl",r"models/MNVBC_model.pkl",r"models/GBC_model.pkl",r"models/DT_model.pkl"]
        st.subheader("Input the News content below")
        predictions = []
        user_input = st.text_area("Enter your news content here", "Some news",height=200)
        if st.button("predict"):
            for model in model_file_list:
                filename = model
                model = pickle.load(open(filename, "rb"))
                prediction = model.predict([user_input])[0]
                predictions.append(prediction)

            dict_prediction = {"Models":model_list,"predictions":predictions}
            df = pd.DataFrame(dict_prediction)

            num_values = df["predictions"].value_counts().tolist()
            num_labels = df["predictions"].value_counts().keys().tolist()

            dict_values = {"true/fake":num_labels,"values":num_values}
            df_prediction = pd.DataFrame(dict_values)
            fig = px.pie(df_prediction, values='values', names='true/fake')
            fig.update_layout(title_text="Comparision between all 7 models: Prediction proportion between True/Fake")
            st.plotly_chart(fig, use_container_width=True)
