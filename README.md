# Rainfall Prediction Using Machine Learning

## Objective:

The objective of this project is to predict whether it will rain or not based on weather features such as temperature, humidity, and other relevant factors. This is a binary classification problem where the outcome is categorized as Rain or No Rain. The model estimates the probability of rainfall and classifies it accordingly.
Additionally, users can enter their own values, and the entire implementation is designed with Streamlit to provide a user-friendly interface.

## Models Used:

The following Machine Learning algorithms are utilized for prediction:
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost Classifier

## Comparision of models:

The Accuracy of Logistic Regression, Support Vector Machine(SVM),XGBoost Classifier reveals that Logistic Regression outperforming SVM and XGBoost in accuracy, achieving the highest accuracy rate among the three models

## Structure of project :

- **project/**
  - **pages/**
    - `1_Data_visualisation.py`
    - `2_check_prediction.py`
  - `Homepage.py`
  - `Rainfall.csv`

## Technologies used

- Python
- Pandas, NumPy (for data preprocessing)
- Scikit-learn, XGBoost (for ML models)
- Matplotlib, Seaborn (for visualization)
- Streamlit (for web-based UI)

## Installation and Usage

### **1. Install Streamlit**  
Run the following command to install Streamlit:  
```bash
pip install streamlit
```
### **2.Run the application**
Execute the following command to launch the application:
```bash
streamlit run Homepage.py
```
