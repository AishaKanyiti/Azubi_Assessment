import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load trained model
model = joblib.load('xgb_subscription_model.pkl')

st.title("📊💵 Bank Term Deposit Subscription Predictor")
st.write("Predict whether a customer will subscribe to a term deposit based on their profile.")

# --- Human-readable option mappings ---
job_options = {
    "Admin": 0, "Blue-collar": 1, "Entrepreneur": 2, "Housemaid": 3,
    "Management": 4, "Retired": 5, "Self-employed": 6, "Services": 7,
    "Student": 8, "Technician": 9, "Unemployed": 10, "Unknown": 11
}

marital_options = {"Married": 0, "Single": 1, "Divorced": 2}

education_options = {"Primary": 0, "Secondary": 1, "Tertiary": 2, "Unknown": 3}

contact_options = {"Cellular": 0, "Telephone": 1, "Unknown": 2}

poutcome_options = {"Failure": 0, "Other": 1, "Success": 2, "Unknown": 3}

age_group_options = {
    "18–25": 0, "26–35": 1, "36–45": 2, "46–60": 3, "60+": 4
}

balance_group_options = {
    "Low": 0, "Medium": 1, "High": 2, "Very High": 3
}

month_options = {
    "January": 0, "February": 1, "March": 2, "April": 3,
    "May": 4, "June": 5, "July": 6, "August": 7,
    "September": 8, "October": 9, "November": 10, "December": 11
}

# --- Input fields ---
job = st.selectbox("Job", job_options.keys())
marital = st.selectbox("Marital Status", marital_options.keys())
education = st.selectbox("Education Level", education_options.keys())
default = st.selectbox("Has Credit in Default?", ["No", "Yes"])
balance = st.number_input("Account Balance", value=0)
housing = st.selectbox("Has Housing Loan?", ["No", "Yes"])
loan = st.selectbox("Has Personal Loan?", ["No", "Yes"])
contact = st.selectbox("Contact Communication Type", contact_options.keys())
month = st.selectbox("Last Contact Month", month_options.keys())
duration = st.number_input("Last Contact Duration (seconds)", value=180)
campaign = st.number_input("Number of Contacts in This Campaign", value=1)
previous = st.number_input("Number of Previous Contacts", value=0)
poutcome = st.selectbox("Previous Marketing Outcome", poutcome_options.keys())
was_previously_contacted = st.selectbox("Was Previously Contacted?", ["No", "Yes"])
age_group = st.selectbox("Age Group", age_group_options.keys())
balance_group = st.selectbox("Balance Category", balance_group_options.keys())

# --- Encode input values ---
input_data = pd.DataFrame([[
    job_options[job],
    marital_options[marital],
    education_options[education],
    1 if default == "Yes" else 0,
    balance,
    1 if housing == "Yes" else 0,
    1 if loan == "Yes" else 0,
    contact_options[contact],
    month_options[month],
    duration,
    campaign,
    previous,
    poutcome_options[poutcome],
    1 if was_previously_contacted == "Yes" else 0,
    age_group_options[age_group],
    balance_group_options[balance_group]
]], columns=[
    'job', 'marital', 'education', 'default', 'balance', 'housing',
    'loan', 'contact', 'month', 'duration', 'campaign',
    'previous', 'poutcome', 'was_previously_contacted',
    'age_group', 'balance_group'
])

# --- Predict and show result ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "✅ Subscribed" if prediction == 1 else "❌ Not Subscribed"
    st.success(f"The model predicts: {result}")
