import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import base64
from ydata_profiling import ProfileReport

# Load the trained models
model_v1 = joblib.load("random_forest_model_v1.pkl")
model_v2 = joblib.load("random_forest_model_v2.pkl")

def preprocess_data(data):
    data = data.drop(['Residence_City', 'Residence_State', 'Applicant_ID'], axis=1)
    # Handle null values
    if data.isnull().any().any():
        # Separate categorical and numerical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        numerical_cols = data.select_dtypes(include=['number']).columns
        
        # Handle null values for categorical columns
        if categorical_cols.any():
            data[categorical_cols] = data[categorical_cols].fillna('unknown')
        
        # Handle null values for numerical columns
        if numerical_cols.any():
            imputer_numerical = SimpleImputer(strategy='mean')
            data[numerical_cols] = imputer_numerical.fit_transform(data[numerical_cols])
    
    # Encode the "House_Ownership" column using LabelEncoder
    label_encoder = LabelEncoder()

    categorical_features = ['Marital_Status', 'House_Ownership', 'Vehicle_Ownership(car)', 'Occupation']

    for feature in categorical_features:
        data[feature] = label_encoder.fit_transform(data[feature])

    
    # Select features (X) and target variable (y)
    X = data.drop('Loan_Default_Risk', axis=1)
    y = data['Loan_Default_Risk']
    
    return X, y

def scale_data(X, scaler):
    return scaler.transform(X)

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, cm

# Main function for Streamlit app
def main():
    st.title('Welcome to the Loan Approval Prediction WebApp')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Preprocess the data
        X, y = preprocess_data(data)

        # Add dropdown menu for model version selection
        model_version = st.selectbox("Choose Model Version", ("Version 1", "Version 2"))

        # Add submit button
        if st.button("Submit"):
            # Scale the data
            scaler = joblib.load("scaler.pkl")
            X_scaled = scale_data(X, scaler)

            # Make predictions and calculate evaluation metrics
            if model_version == "Version 1":
                st.write("Using Version 1 Model")
                predictions = model_v1.predict(X_scaled)
            elif model_version == "Version 2":
                st.write("Using Version 2 Model")
                predictions = model_v2.predict(X_scaled)

            # Calculate evaluation metrics
            accuracy, precision, recall, f1, cm = calculate_metrics(y, predictions)

            # Display evaluation metrics
            st.write("Evaluation Metrics:")
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Precision: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1-Score: {f1}")
            st.write("Confusion Matrix:")
            st.write(cm)

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y, predictions)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            st.pyplot(plt)
            
            # Generate EDA report and save as HTML file
            profile = ProfileReport(data, title='Pandas Profiling Report', explorative=True)
            profile_path = "EDA.html"
            profile.to_file(profile_path)

            # Option to download EDA HTML file
            eda_html_file = open(profile_path, "rb").read()
            href = f'<a href="data:file/html;base64,{base64.b64encode(eda_html_file).decode()}" download="EDA.html">Download EDA HTML File</a>'
            st.markdown(href, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
