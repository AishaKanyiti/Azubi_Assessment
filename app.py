streamlit
pandas
numpy
scikit-learn
xgboost
joblib


# Load the trained model
model = joblib.load('xgb_subscription_model.pkl')

st.title("Bank Marketing Prediction App")
st.write("Predict whether a customer will subscribe to a term deposit.")

# Input fields
job = st.selectbox("Job", list(range(0, 12)))  # Replace with your encoded values or a mapping
marital = st.selectbox("Marital Status", list(range(0, 3)))
education = st.selectbox("Education", list(range(0, 4)))
default = st.selectbox("Default", [0, 1])
balance = st.number_input("Balance", value=0)
housing = st.selectbox("Housing Loan", [0, 1])
loan = st.selectbox("Personal Loan", [0, 1])
contact = st.selectbox("Contact Communication Type", list(range(0, 3)))
day = st.number_input("Last Contact Day of the Month", value=15)
month = st.selectbox("Last Contact Month", list(range(0, 12)))
duration = st.number_input("Last Contact Duration (seconds)", value=180)
campaign = st.number_input("Number of Contacts in this Campaign", value=1)
previous = st.number_input("Number of Contacts Before this Campaign", value=0)
poutcome = st.selectbox("Previous Outcome", list(range(0, 4)))
was_previously_contacted = st.selectbox("Was Previously Contacted", [0, 1])
age_group = st.selectbox("Age Group", list(range(0, 5)))  # e.g., 0: 18–25, 1: 26–35...
balance_group = st.selectbox("Balance Group", list(range(0, 4)))  # e.g., 0: Low, 1: Med...

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
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Subscribed" if prediction == 1 else "Not Subscribed"
    st.success(f"The model predicts: {result}")
