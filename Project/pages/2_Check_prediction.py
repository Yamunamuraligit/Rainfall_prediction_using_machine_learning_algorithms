import streamlit as st
import numpy as np

def get_user_input():
    """
    This function gathers user inputs using Streamlit widgets.
    """
    pressure = st.number_input("Enter the pressure")
    maxtemp = st.number_input("Enter the max temperature")
    temperature = st.number_input("Enter the temperature")
    mintemp = st.number_input("Enter the min temperature")
    dewpoint = st.number_input("Enter the dewpoint")
    humidity = st.number_input("Enter the humidity")
    cloud = st.number_input("Enter the cloud cover")
    sunshine = st.number_input("Enter the sunshine")
    winddirection = st.number_input("Enter the wind direction")
    windspeed = st.number_input("Enter the wind speed")

    user_input = np.array([pressure, maxtemp, temperature, mintemp, dewpoint,
                           humidity, cloud, sunshine, winddirection, windspeed]).reshape(1, -1)

    return user_input

def predict_rainfall(model, user_input):
    """
    This function makes a prediction using the trained model and user input.
    """
    prediction = model.predict(user_input)
    return "Rainfall" if prediction == 1 else "No Rainfall"

def main():
    st.title("Rainfall Prediction")

    # Get user inputs
    user_input = get_user_input()


    # Choose which model to use
    model_choice = st.selectbox("Pick the model", ["Logistic Regression", "Support Vector Machine", "XGBoost Classifier"])

    # Load the selected model from session_state
    if model_choice == "Logistic Regression":
        model = st.session_state.get('log_reg_model', None)
    elif model_choice == "Support Vector Machine":
        model = st.session_state.get('svm_model', None)
    elif model_choice == "XGBoost Classifier":
        model = st.session_state.get('xgb_model', None)

    # Check if model is loaded, else show error
    if model is None:
        st.error("The selected model is not available. Please train and save the model first.")
        return

    # Make prediction using the selected model
    if st.button("Predict"):
        result = predict_rainfall(model, user_input)
        st.success(f"The prediction is: {result}")

if __name__ == "__main__":
    main()
