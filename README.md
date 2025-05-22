# Rainfall Prediction Using Machine Learning

## Objective:

The objective of this project is to predict whether it will rain or not based on weather features such as temperature, humidity, and other relevant factors. This is a binary classification problem where the outcome is categorized as Rain or No Rain. The model estimates the probability of rainfall and classifies it accordingly.
Additionally, users can enter their own values, and the entire implementation is designed with Streamlit to provide a user-friendly interface.

## Models Used:

The following Machine Learning algorithms are utilized for prediction:
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost Classifier

## Inputs/Features
The following features are used for predicting rainfall:

- **Pressure**: Atmospheric pressure at the given time.
- **Temperature**: Temperature at the given time.
- **MinTemp**: Minimum temperature during the given period.
- **MaxTemp**: Maximum temperature during the given period.
- **Dewpoint**: Dew point temperature.
- **Humidity**: Humidity level in the air.
- **Cloud**: Cloud coverage during the given period.
- **Sunshine**: Sunshine duration or intensity.
- **Windspeed**: Wind speed at the given time.
- **Wind Direction**: The direction of the wind during the period.


## Comparision of models:

The Accuracy of Logistic Regression, Support Vector Machine(SVM),XGBoost Classifier reveals that XGBoost Classifier outperforming than Logistic Regression and SVM in accuracy, achieving the highest accuracy rate among the three models

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
### **2. Run the application**
Execute the following command to launch the application:
```bash
streamlit run Homepage.py
```
### **3. Homepage**
This command will open the Rainfall Prediction Homepage, where you can check the inputs and view the accuracy of each model.

### **4. Data Visualisation**
On the left-hand side, you will find the Data Visualization section, which shows the training and testing accuracy of all three models.

### **5. Check Prediction**
In the Check Prediction section, users can enter their own input values and select a model to predict whether it will Rain or Not Rain, along with the corresponding accuracy.

## Result

### **Model Accuracy**  
The models were evaluated based on **accuracy**. Below is a comparison of their performance:  

| Model                  | Accuracy |
|------------------------|----------|
| **Logistic Regression** | 85%      |
| **SVM**                 | 88%      |
| **XGBoost**             | 91%      |
