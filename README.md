Rainfall Prediction Using Machine Learning

Objective:

The objective of this project is to predict whether it will rain or not based on weather features such as temperature, humidity, and other relevant factors. This is a binary classification problem where the outcome is categorized as Rain or No Rain. The model estimates the probability of rainfall and classifies it accordingly.
Additionally, users can enter their own values, and the entire implementation is designed with Streamlit to provide a user-friendly interface.

Techniques Used:

The following Machine Learning algorithms are utilized for prediction:
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost Classifier

Comparision of models:

The Accuracy of Logistic Regression, Support Vector Machine(SVM),XGBoost Classifier reveals that Logistic Regression outperforming SVM and XGBoost in accuracy, achieving the highest accuracy rate among the three models

Structure of project :

rainfall-prediction/
│── data/                     # Directory for dataset files
│   ├── rainfall_data.csv      # Raw dataset file
│   ├── processed_data.csv     # Preprocessed dataset
│── models/                    # Trained models storage
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   ├── xgboost_model.pkl
│── notebooks/                 # Jupyter notebooks for EDA and model training
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│── src/                       # Source code
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── train_model.py         # Model training script
│   ├── predict.py             # Prediction script
│── app/                       # Streamlit application
│   ├── app.py                 # Main Streamlit app
│── requirements.txt           # Python dependencies
│── README.md                  # Project documentation
│── LICENSE                    # License file
│── .gitignore                 # Files to ignore in Git













|            |--------> 1_Data_visualisation.py
|            |--------> 2_check_prediction.py
|--------> Homepage.py
|--------> Rainfall.csv
