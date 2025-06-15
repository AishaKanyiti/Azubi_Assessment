{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "724e2c7d-c943-4e48-a6da-499464819737",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load('xgb_subscription_model.pkl')\n",
    "\n",
    "st.title(\"Bank Marketing Prediction App\")\n",
    "st.write(\"Predict whether a customer will subscribe to a term deposit.\")\n",
    "\n",
    "# Input fields\n",
    "job = st.selectbox(\"Job\", list(range(0, 12)))  # Replace with your encoded values or a mapping\n",
    "marital = st.selectbox(\"Marital Status\", list(range(0, 3)))\n",
    "education = st.selectbox(\"Education\", list(range(0, 4)))\n",
    "default = st.selectbox(\"Default\", [0, 1])\n",
    "balance = st.number_input(\"Balance\", value=0)\n",
    "housing = st.selectbox(\"Housing Loan\", [0, 1])\n",
    "loan = st.selectbox(\"Personal Loan\", [0, 1])\n",
    "contact = st.selectbox(\"Contact Communication Type\", list(range(0, 3)))\n",
    "day = st.number_input(\"Last Contact Day of the Month\", value=15)\n",
    "month = st.selectbox(\"Last Contact Month\", list(range(0, 12)))\n",
    "duration = st.number_input(\"Last Contact Duration (seconds)\", value=180)\n",
    "campaign = st.number_input(\"Number of Contacts in this Campaign\", value=1)\n",
    "previous = st.number_input(\"Number of Contacts Before this Campaign\", value=0)\n",
    "poutcome = st.selectbox(\"Previous Outcome\", list(range(0, 4)))\n",
    "was_previously_contacted = st.selectbox(\"Was Previously Contacted\", [0, 1])\n",
    "age_group = st.selectbox(\"Age Group\", list(range(0, 5)))  # e.g., 0: 18–25, 1: 26–35...\n",
    "balance_group = st.selectbox(\"Balance Group\", list(range(0, 4)))  # e.g., 0: Low, 1: Med...\n",
    "\n",
    "# Prepare input\n",
    "input_data = pd.DataFrame([[\n",
    "    job, marital, education, default, balance, housing, loan, contact,\n",
    "    day, month, duration, campaign, previous, poutcome,\n",
    "    was_previously_contacted, age_group, balance_group\n",
    "]], columns=[\n",
    "    'job', 'marital', 'education', 'default', 'balance', 'housing',\n",
    "    'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'previous',\n",
    "    'poutcome', 'was_previously_contacted', 'age_group', 'balance_group'\n",
    "])\n",
    "\n",
    "# Predict\n",
    "if st.button(\"Predict\"):\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    result = \"Subscribed\" if prediction == 1 else \"Not Subscribed\"\n",
    "    st.success(f\"The model predicts: {result}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
