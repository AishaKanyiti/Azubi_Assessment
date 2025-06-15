import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load trained model
model = joblib.load('xgb_subscription_model.pkl')

st.title("Bank Term Deposit Subscription Predictor")
st.write("Use this tool to predict if a client will subscribe to a term deposit based on their details.")

# Mappings for categorical inputs
job_mapping = {
    "Admin": 0, "Blue-collar": 1, "Entrepreneur": 2, "Housemaid": 3,
    "Management": 4, "Retired": 5, "Self-employed": 6, "Services": 7,
    "Student": 8, "Technician": 9, "Unemployed": 10, "Unknown": 11
}
marital_mapping = {"Married": 0, "Single": 1, "Divorced": 2}
education_mapping = {"Primary": 0, "Secondary": 1, "Tertiary": 2, "Unknown": 3}
contact_mapping = {"Cellular": 0, "Telephone": 1, "Unknown": 2}
month_mapping = {
    "January": 0, "February": 1, "March": 2, "April": 3,
    "May": 4, "June": 5, "July": 6, "August": 7,
    "September": 8, "October": 9, "November": 10, "December": 11
}
poutcome_mapping = {"Failure": 0, "Other": 1, "Success": 2, "Unknown": 3}
age_group_mapping = {"18–25": 0, "26–35": 1, "36–45": 2, "46–60": 3, "60+": 4}
balance_group_mapping = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}

# User inputs
job = job_mapping[st.selectbox("Job", list(job_mapping.keys()))]
marital = marital_mapping[st.selectbox("Marital Status", list(marital_mapping.keys()))]
education = education_mapping[st.selectbox("Education Level", list(education_mapping.keys()))]
default = st.selectbox("Has Credit Default?", ["No", "Yes"])
default = 1 if default == "Yes" else 0
balance = st.number_input("Account Balance (€)", value=0)
housing = st.selectbox("Has Housing Loan?", ["No", "Yes"])
housing = 1 if housing == "Yes" else 0
loan = st.selectbox("Has Personal Loan?", ["No", "Yes"])
loan = 1 if loan == "Yes" else 0
contact = contact_mapping[st.selectbox("Contact Method", list(contact_mapping.keys()))]
day = st.number_input("Last Contact Day of Month", value=15, min_value=1, max_value=31)
month = month_mapping[st.selectbox("Last Contact Month", list(month_mapping.keys()))]
duration = st.number_input("Contact Duration (seconds)", value=180)
campaign = st.number_input("Campaign Contacts", value=1, min_value=1)
previous = st.number_input("Previous Contacts", value=0)
poutcome = poutcome_mapping[st.selectbox("Previous Campaign Outcome", list(poutcome_mapping.keys()))]
was_previously_contacted = st.selectbox("Previously Contacted?", ["No", "Yes"])
was_previously_contacted = 1 if was_previously_contacted == "Yes" else 0
age_group = age_group_mapping[st.selectbox("Age Group", list(age_group_mapping.keys()))]
balance_group = balance_group_mapping[st.selectbox("Balance Category", list(balance_group_mapping.keys()))]

# Prepare input
input_data = pd.DataFrame([[
    job, marital, education, default, balance, housing, loan, contact,
    day, month, duration, campaign, previous, poutcome,
    was_previously_contacted, age_group, balance_group
]], columns=[
    'job', 'marital', 'education', 'default', 'balance', 'housing',
    'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'previous',
    'poutcome', 'was_previously_contacted', 'age_group', 'balance_group'
])

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    result = "✅ Subscribed" if prediction == 1 else "❌ Not Subscribed"
   st.success(f"The model predicts: {result}")
 
