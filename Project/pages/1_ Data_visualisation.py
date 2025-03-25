import streamlit as st 

st.title("Data visualisation")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.info("Logistic Regression")
Train =st.session_state['log_reg_train_accuracy']
Test = st.session_state['log_reg_test_accuracy']

models = pd.DataFrame({'Models':['Training','Testing'],
                        'Values': [Train,Test]})
models.set_index('Models', inplace=True)
st.bar_chart(models)

st.info("Support Vector Machine")
Train1 = st.session_state['svm_train_accuracy']
Test1 = st.session_state['svm_test_accuracy']

models = pd.DataFrame({'Models':['Training','Testing'],
                        'Values': [Train1,Test1]})
models.set_index('Models', inplace=True)
st.bar_chart(models)

st.info("XGBoostClassifier")
Train2 = st.session_state['xgb_train_accuracy']
Test2 = st.session_state['xgb_test_accuracy']

models = pd.DataFrame({'Models':['Training','Testing'],
                        'Values': [Train2,Test2]})
models.set_index('Models', inplace=True)
st.bar_chart(models)

st.info("Comparision of the Models")

compare = pd.DataFrame({'Models' : ['Logistic Regression','Support vactor machine','XGBoost Classifier'],
                        'Values' :[Test,Test1,Test2]})
compare.set_index('Models',inplace =True)
st.bar_chart(compare)
