import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load the trained XGBoost model
model = joblib.load('xgb_subscription_model.pkl')

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="centered")
st.title("💼 Bank Term Deposit Subscription Predictor")
st.write("Fill in the client's details to predict the likelihood of subscribing to a term deposit.")

# Mapping dictionaries
job_map = {
    'Admin': 0, 'Blue-collar': 1, 'Entrepreneur': 2, 'Housemaid': 3,
    'Management': 4, 'Retired': 5, 'Self-employed': 6, 'Services': 7,
    'Student': 8, 'Technician': 9, 'Unemployed': 10, 'Unknown': 11
}

marital_map = {'Divorced': 0, 'Married': 1, 'Single': 2}
education_map = {'Primary': 0, 'Secondary': 1, 'Tertiary': 2, 'Unknown': 3}
contact_map = {'Cellular': 0, 'Telephone': 1, 'Unknown': 2}
month_map = {
    'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
    'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
}
poutcome_map = {'Failure': 0, 'Other': 1, 'Success': 2, 'Unknown': 3}
age_group_map = {'18–25': 0, '26–35': 1, '36–45': 2, '46–55': 3, '56–65': 4, '66+': 5}
balance_group_map = {'Debt': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", list(job_map.keys()))
marital = st.selectbox("Marital Status", list(marital_map.keys()))
education = st.selectbox("Education Level", list(education_map.keys()))
default = st.selectbox("Has Credit in Default?", ['No', 'Yes'])
balance = st.number_input("Account Balance", value=0)
housing = st.selectbox("Has Housing Loan?", ['No', 'Yes'])
loan = st.selectbox("Has Personal Loan?", ['No', 'Yes'])
contact = st.selectbox("Contact Communication Type", list(contact_map.keys()))
month = st.selectbox("Last Contact Month", list(month_map.keys()))
duration = st.number_input("Last Contact Duration (seconds)", value=180)
campaign = st.number_input("Number of Contacts in This Campaign", value=1)
previous = st.number_input("Number of Previous Contacts", value=0)
poutcome = st.selectbox("Previous Marketing Outcome", list(poutcome_map.keys()))
was_previously_contacted = st.selectbox("Was Previously Contacted?", ['No', 'Yes'])
age_group = st.selectbox("Age Group", list(age_group_map.keys()))
balance_group = st.selectbox("Balance Category", list(balance_group_map.keys()))

# Encode inputs
input_data = pd.DataFrame([[
    age,
    job_map[job],
    marital_map[marital],
    education_map[education],
    1 if default == 'Yes' else 0,
    balance,
    1 if housing == 'Yes' else 0,
    1 if loan == 'Yes' else 0,
    contact_map[contact],
    month_map[month],
    duration,
    campaign,
    previous,
    poutcome_map[poutcome],
    1 if was_previously_contacted == 'Yes' else 0,
    age_group_map[age_group],
    balance_group_map[balance_group]
]], columns=[
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
    'loan', 'contact', 'month', 'duration', 'campaign', 'previous',
    'poutcome', 'was_previously_contacted', 'age_group', 'balance_group'
])

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    result = "✅ Subscribed" if prediction == 1 else "❌ Not Subscribed"
    st.success(f"The model predicts: **{result}**")

