import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Set a wide layout for the app
st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load the model and scaler ---
@st.cache_resource
def load_resources():
    try:
        # Load the trained model
        with open('model_credit_risk.pkl', 'rb') as model_file:
            model = joblib.load(model_file)

        # Load the numerical variables scaler
        with open('numerical_vars_scaler.pkl', 'rb') as scaler_file:
            scaler = joblib.load(scaler_file)

        # Load the data for insights page
        data_clean = pd.read_csv('cr_loan_clean.csv')

        return model, scaler, data_clean
    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure 'model_credit_risk.pkl', 'numerical_vars_scaler.pkl', and 'cr_loan_clean.csv' are in the same directory.")
        return None, None, None

model, scaler, data_clean = load_resources()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: transform 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    .st-emotion-cache-1av5h81 {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1a1a1a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h2 {
        color: #333333;
    }
    .stMarkdown p {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #555555;
    }
    .result-box-green {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #c3e6cb;
    }
    .result-box-red {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Credit Risk", "Data Insights"])

if page == "Home":
    st.title("Welcome to the Credit Risk Prediction App")
    st.write("---")
    st.header("About This Application")
    st.markdown("""
        <p>
            This interactive application leverages a sophisticated machine learning model to predict the credit risk of loan applicants. By entering specific financial and personal information, you can get an immediate assessment of the likelihood of a loan default.
        </p>
        <p>
            The underlying model, a **CatBoost Classifier**, was trained on a comprehensive dataset of past loan applications. The app's pipeline includes data preprocessing steps, such as one-hot encoding for categorical variables and scaling of numerical features, ensuring accurate predictions.
        </p>
        <p>
            This tool is designed to provide valuable insights for financial institutions and individuals alike, aiding in decision-making and risk management.
        </p>
    """, unsafe_allow_html=True)
    st.image("https://placehold.co/800x400/007bff/ffffff?text=Credit+Risk+Dashboard", caption="Credit Risk Dashboard")
    st.subheader("How It Works")
    st.markdown("""
        1.  **Input Data**: Provide the applicant's details such as income, loan amount, and credit history.
        2.  **Preprocessing**: The app automatically cleans and prepares the data for the model.
        3.  **Prediction**: The model calculates the probability of default.
        4.  **Result**: The app displays a clear and concise prediction, helping you understand the risk involved.
    """)

elif page == "Predict Credit Risk":
    if model is None or scaler is None:
        st.stop()

    st.title("Predict Credit Risk")
    st.write("---")
    st.subheader("Enter the loan applicant's details below:")

    # Define the categorical and numerical feature columns
    numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    # Define the unique values for categorical features
    home_ownership_options = ['MORTGAGE', 'OTHER', 'OWN', 'RENT']
    loan_intent_options = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
    loan_grade_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    default_on_file_options = ['N', 'Y']

    # Create input forms
    with st.form("input_form"):
        # Use two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Financial & Personal Information")
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            person_income = st.number_input("Annual Income ($)", min_value=0, value=75000)
            person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=60.0, value=5.0)
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=15000)

        with col2:
            st.markdown("### Loan Details & History")
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, format="%.2f")
            loan_percent_income = st.number_input("Loan/Income (%)", min_value=0.0, max_value=1.0, value=0.2, format="%.2f")
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
            person_home_ownership = st.selectbox("Home Ownership", options=home_ownership_options)
            loan_intent = st.selectbox("Loan Intent", options=loan_intent_options)
            loan_grade = st.selectbox("Loan Grade", options=loan_grade_options)
            cb_person_default_on_file = st.selectbox("Credit Default History", options=default_on_file_options)

        predict_button = st.form_submit_button("Predict")

    if predict_button:
        # Create a dataframe from user inputs
        input_data = pd.DataFrame([{
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_length': person_emp_length,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'person_home_ownership': person_home_ownership,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'cb_person_default_on_file': cb_person_default_on_file
        }])

        # One-hot encode categorical features
        # We need to create a full dummy dataframe to ensure all columns exist
        one_hot_cols = [f"person_home_ownership_{ho}" for ho in home_ownership_options] + \
                       [f"loan_intent_{li}" for li in loan_intent_options] + \
                       [f"loan_grade_{lg}" for lg in loan_grade_options] + \
                       [f"cb_person_default_on_file_{df}" for df in default_on_file_options]

        dummy_df = pd.DataFrame(0, index=input_data.index, columns=one_hot_cols)

        # Set the value to 1 for the selected options
        dummy_df[f'person_home_ownership_{person_home_ownership}'] = 1
        dummy_df[f'loan_intent_{loan_intent}'] = 1
        dummy_df[f'loan_grade_{loan_grade}'] = 1
        dummy_df[f'cb_person_default_on_file_{cb_person_default_on_file}'] = 1

        # Drop the original categorical columns and concatenate the new ones
        input_data = input_data.drop(columns=categorical_cols)
        input_data = pd.concat([input_data, dummy_df], axis=1)

        # Scale numerical features
        scaled_numerical_data = scaler.transform(input_data[numerical_cols])
        input_data[numerical_cols] = scaled_numerical_data

        # Reorder columns to match model's training order
        model_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                          'cb_person_cred_hist_length', 'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
                          'person_home_ownership_OWN', 'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
                          'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL',
                          'loan_intent_VENTURE', 'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E',
                          'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_N', 'cb_person_default_on_file_Y']

        input_data = input_data[model_features]

        # Make prediction
        prediction_proba = model.predict_proba(input_data)[0]
        prediction = np.argmax(prediction_proba)

        st.write("---")
        st.subheader("Prediction Result")

        if prediction == 0:
            st.markdown(f'<div class="result-box-green"><h3>✅ Loan Approved: Low Risk of Default</h3><p>The model predicts a very low probability of default ({prediction_proba[1]:.2%} chance). This loan is recommended for approval.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box-red"><h3>❌ Loan Declined: High Risk of Default</h3><p>The model predicts a significant probability of default ({prediction_proba[1]:.2%} chance). This loan carries a high risk and is not recommended for approval.</p></div>', unsafe_allow_html=True)
        st.write("For an in-depth analysis of the factors contributing to this prediction, you may need to consult the full model details.")


elif page == "Data Insights":
    if data_clean is None:
        st.stop()

    st.title("Credit Risk Data Insights")
    st.write("---")

    st.subheader("Loan Status Distribution")
    loan_status_counts = data_clean['loan_status'].value_counts()
    st.bar_chart(loan_status_counts, use_container_width=True)
    st.write(f"This chart shows the distribution of loan outcomes in the dataset. A value of **0** indicates no default, while **1** indicates a default.")

    st.subheader("Credit Default by Loan Grade")
    loan_grade_default = data_clean.groupby('loan_grade')['loan_status'].mean().sort_values()
    st.bar_chart(loan_grade_default, use_container_width=True)
    st.write("This plot illustrates the average default rate for each loan grade, from A (lowest risk) to G (highest risk).")

    st.subheader("Loan Amount vs. Annual Income")
    st.scatter_chart(data_clean, x='person_income', y='loan_amnt', color='loan_status')
    st.write("This scatter plot visualizes the relationship between the loan amount and a person's annual income. Points are colored by loan status to highlight patterns.")

    st.subheader("Top 20 Risky Loans (by Income/Loan %)")
    risky_loans = data_clean.sort_values(by='loan_percent_income', ascending=False).head(20)
    st.dataframe(risky_loans)
    st.write("This table shows the top 20 loans with the highest ratio of loan amount to annual income, which is often a strong indicator of risk.")
