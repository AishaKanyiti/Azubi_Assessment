import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgb_subscription_model.pkl")

# Mapping dictionaries (based on how you encoded the training data)
job_map = {
    "Admin": 0, "Blue-collar": 1, "Entrepreneur": 2, "Housemaid": 3, "Management": 4,
    "Retired": 5, "Self-employed": 6, "Services": 7, "Student": 8, "Technician": 9, "Unemployed": 10, "Unknown": 11
}
marital_map = {"Married": 0, "Single": 1, "Divorced": 2}
education_map = {"Primary": 0, "Secondary": 1, "Tertiary": 2, "Unknown": 3}
contact_map = {"Cellular": 0, "Telephone": 1, "Unknown": 2}
month_map = {
    "January": 0, "February": 1, "March": 2, "April": 3, "May": 4, "June": 5,
    "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11
}
poutcome_map = {"Failure": 0, "Other": 1, "Success": 2, "Unknown": 3}
age_group_map = {"18–25": 0, "26–35": 1, "36–45": 2, "46–55": 3, "56–65": 4, "66+": 5}
balance_group_map = {"Debt": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4}

st.title("📈 Term Deposit Subscription Prediction")
st.write("Enter client details to predict term deposit subscription.")

# App Inputs (friendly text, mapped to numerical values)
job = st.selectbox("Job", list(job_map.keys()))
marital = st.selectbox("Marital Status", list(marital_map.keys()))
education = st.selectbox("Education", list(education_map.keys()))
default = st.selectbox("Has Credit in Default?", ["Yes", "No"])
balance = st.number_input("Account Balance", value=0)
housing = st.selectbox("Has Housing Loan?", ["Yes", "No"])
loan = st.selectbox("Has Personal Loan?", ["Yes", "No"])
contact = st.selectbox("Contact Type", list(contact_map.keys()))
month = st.selectbox("Last Contact Month", list(month_map.keys()))
duration = st.number_input("Last Contact Duration (seconds)", value=100)
campaign = st.number_input("Number of Contacts in This Campaign", value=1)
previous = st.number_input("Number of Previous Contacts", value=0)
poutcome = st.selectbox("Previous Campaign Outcome", list(poutcome_map.keys()))
was_contacted = st.selectbox("Was Previously Contacted?", ["Yes", "No"])
age_group = st.selectbox("Age Group", list(age_group_map.keys()))
balance_group = st.selectbox("Balance Category", list(balance_group_map.keys()))

# Prepare input for prediction
input_data = pd.DataFrame([[
    job_map[job], marital_map[marital], education_map[education],
    1 if default == "Yes" else 0, balance,
    1 if housing == "Yes" else 0, 1 if loan == "Yes" else 0,
    contact_map[contact], month_map[month], duration,
    campaign, previous, poutcome_map[poutcome],
    1 if was_contacted == "Yes" else 0,
    age_group_map[age_group], balance_group_map[balance_group]
]], columns=[
    'job', 'marital', 'education', 'default', 'balance', 'housing',
    'loan', 'contact', 'month', 'duration', 'campaign', 'previous',
    'poutcome', 'was_previously_contacted', 'age_group', 'balance_group'
])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "✅ Subscribed" if prediction == 1 else "❌ Not Subscribed"
    st.success(f"Prediction: {result}")
