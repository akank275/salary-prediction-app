import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# Load model
model = joblib.load("salary_model.pkl")

# Load dataset
df = pd.read_csv("cleaned_salary_data.csv")

# Title
st.title("💰 Employee Salary Prediction App")

st.write(
    "Predict salaries using Machine Learning"
)

# Sidebar
st.sidebar.header("Enter Employee Details")

experience_level = st.sidebar.selectbox(
    "Experience Level",
    sorted(df["experience_level"].unique())
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    sorted(df["employment_type"].unique())
)

job_title = st.sidebar.selectbox(
    "Job Title",
    sorted(df["job_title"].unique())
)

employee_residence = st.sidebar.selectbox(
    "Employee Residence",
    sorted(df["employee_residence"].unique())
)

remote_ratio = st.sidebar.selectbox(
    "Remote Ratio",
    sorted(df["remote_ratio"].unique())
)

company_location = st.sidebar.selectbox(
    "Company Location",
    sorted(df["company_location"].unique())
)

company_size = st.sidebar.selectbox(
    "Company Size",
    sorted(df["company_size"].unique())
)

# Input dataframe
input_data = pd.DataFrame({
    "experience_level": [experience_level],
    "employment_type": [employment_type],
    "job_title": [job_title],
    "employee_residence": [employee_residence],
    "remote_ratio": [remote_ratio],
    "company_location": [company_location],
    "company_size": [company_size]
})

# Prediction
if st.button("Predict Salary"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Salary: ${prediction[0]:,.2f}"
    )

# Dataset preview
st.subheader("Dataset Preview")

st.dataframe(df.head())

# Salary distribution
st.subheader("Salary Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    df["salary"],
    bins=30,
    kde=True,
    ax=ax
)

st.pyplot(fig)

# Correlation heatmap
st.subheader("Feature Correlation")

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)
