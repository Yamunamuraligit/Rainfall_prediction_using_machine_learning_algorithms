import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title = "Rainfall prediction"
)
st.title("Rainfall Prediction Using Various Algorithms")
st.sidebar.success("select a above page")
st.info("Dataset preview")
df = pd.read_csv('Rainfall.csv')
st.write(df.head())

st.info("Total number of columns and rows")
df.shape

st.info("Total columns")
df.rename(str.strip, axis='columns', inplace=True) # To remove the space before the column name
st.write(df.columns)
st.info("Count null values")
st.write(df.isnull().sum())
st.info("Fill nulls, Recount missing")
for col in df.columns: # finding the null values and their are replaced with the mean value
    if df[col].isnull().sum() > 0:
        val = df[col].mean()
        df[col] = df[col].fillna(val)

st.write(df.isnull().sum())
st.info("Counting the Rainfall Responses")
st.write(df['rainfall'].value_counts()) # value_counts is a function,
                                    # it return the frequncy count of Yes or No in the column
st.info("Exclude the day and rainfall")
features = df.drop(['day', 'rainfall'], axis=1)#we are removing the day and rainfall in the features
st.write(features)
st.info("Convert the labels(yes,no) to binary(1,0)")
df.replace({'yes':1, 'no':0}, inplace=True)#Replacing the yes with 1 and No with 0
target = df.rainfall# The rainfall column is stored in the target
st.write(target)

st.write("Size of the target",target.shape)

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2)#split the model into train and test.The size of size model is 20%
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X, Y = ros.fit_resample(X_train, Y_train) #RandomOverSampler makes the output values are equal.
st.info("X values")
st.write(X)
st.info("Y values")
st.write(Y)

# Logistic Regression
st.info("LogisticRegression")
clf=LogisticRegression()
clf.fit(X, Y)
y_pred = clf.predict(X)
acc_clf1 = accuracy_score(Y ,y_pred)
st.write("Training data accuracy is (in %) : ",acc_clf1*100)

y_pred = clf.predict(X_val)
acc_clf2 = accuracy_score(Y_val,y_pred)
st.write("Testing data accuracy is (in %) : ",acc_clf2*100)

st.session_state['log_reg_train_accuracy'] = acc_clf1*100 # Save training accuracy
st.session_state['log_reg_test_accuracy'] = acc_clf2*100  # Save testing accuracy
st.session_state['log_reg_model'] = clf

# Support vector machine
st.info("Support vector machine")
from sklearn.svm import SVC
svm = SVC()
svm.fit(X, Y)
y_pred = svm.predict(X)
acc_svc1 = accuracy_score(Y, y_pred)
st.write("Training model accuracy (in %):", acc_svc1 * 100)

y_pred = svm.predict(X_val)
acc_svc2 = accuracy_score(Y_val, y_pred)
st.write("Testing model accuracy (in %):", acc_svc2 * 100)

st.session_state['svm_model'] = svm
st.session_state['svm_train_accuracy'] = acc_svc1*100 # Save training accuracy
st.session_state['svm_test_accuracy'] = acc_svc2*100  # Save testing accuracy

# XGBoost Classifier
st.info("XGBoost model")
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X, Y)
y_pred = xgb.predict(X)
acc_xgb1 = accuracy_score(Y, y_pred)
st.write("Training model accuracy (in %) : ", acc_xgb1 * 100)

y_pred = xgb.predict(X_val)
acc_xgb2 = accuracy_score(Y_val,y_pred)
st.write("Testing model accuracy (in %) : ",acc_xgb2 *100)

st.session_state['xgb_model'] = xgb
st.session_state['xgb_train_accuracy'] = acc_xgb1*100 # Save training accuracy
st.session_state['xgb_test_accuracy'] = acc_xgb2*100  # Save testing accuracy
