import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="AI Job Risk Prediction", layout="wide")

page = st.sidebar.radio(" Navigation", ["Predict", "About"])

@st.cache_resource
def load_and_train():
    df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

    le_job = LabelEncoder()
    le_edu = LabelEncoder()
    le_risk = LabelEncoder()

    df['Job_Title'] = le_job.fit_transform(df['Job_Title'])
    df['Education_Level'] = le_edu.fit_transform(df['Education_Level'])
    df['Risk_Category'] = le_risk.fit_transform(df['Risk_Category'])

    X = df.drop(['Risk_Category', 'Automation_Probability_2030'], axis=1)
    y = df['Risk_Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    return df, X, le_job, le_edu, le_risk, rf_model, xgb_model

df, X, le_job, le_edu, le_risk, rf_model, xgb_model = load_and_train()

if page == "About":
    st.title("ℹ About This App")
    st.markdown("""
    ###  About the Dataset & Project

    This application is built using a **simulated dataset** that models how different jobs 
    may be impacted by **AI-driven automation by the year 2030**.  
    The dataset contains 3,000 job entries, each describing:

    - Job title  
    - Required education level  
    - Years of experience  
    - Average salary  
    - AI exposure index  
    - Technology growth factor  
    - Automation probability for 2030  
    - Risk category (Low, Medium, High)  

    The goal of the project is to analyze and predict the **automation risk level of a given job** 
    using machine learning models such as **Random Forest** and **XGBoost**, and to provide 
    interpretability using **SHAP explainability** and feature importance visualization.

    ### 🛠 About the Skill Inputs
    The dataset includes **10 skill indicators**, but these are **generic simulated scores**,  
    not actual real-world skills associated with specific professions.  
    They represent overall *skill importance levels* in the model only.

    **If you are not sure what values to enter for skills,  
    it is recommended to leave them at the default average value (0.5).**  
    This ensures stable predictions without requiring the user to determine exact skill weights.

    ### Purpose of the App
    - Explore and understand which jobs are more vulnerable to automation  
    - Provide explainable ML predictions  
    - Support research and visualization on the future of work  

    **Created by:** *Saja Hamasha* 🤍  
    """)
    st.stop()

st.title("🔮 AI Job Automation Risk Prediction (2030)")
st.write("Predict the automation risk level using Machine Learning.")
st.markdown("---")


# Select Model
model_choice = st.selectbox(
    "Choose Prediction Model:",
    ["Random Forest", "XGBoost"]
)

st.markdown("##  Job Information")

col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox(
        "Job Title:",
        df["Job_Title"].unique(),
        format_func=lambda x: le_job.inverse_transform([x])[0]
    )

    education = st.selectbox(
        "Education Level:",
        df["Education_Level"].unique(),
        format_func=lambda x: le_edu.inverse_transform([x])[0]
    )

with col2:
    avg_salary = st.slider("Average Salary ($):", 20000, 200000, 50000)
    experience = st.slider("Years of Experience:", 0, 40, 5)

st.markdown("##  AI & Technology Factors")

col3, col4 = st.columns(2)

with col3:
    ai_exp = st.slider("AI Exposure Index:", 0.0, 1.0, 0.5)

with col4:
    tech_growth = st.slider("Tech Growth Factor:", 0.5, 3.0, 1.0)

st.markdown("## 🛠 Skills")

skills = []
for i in range(1, 11):
    skills.append(st.slider(f"Skill {i}", 0.0, 1.0, 0.5))

# Create row for prediction
input_data = pd.DataFrame([[
    job_title, avg_salary, experience, education,
    ai_exp, tech_growth, *skills
]], columns=X.columns)


if st.button("🔍 Predict Automation Risk", use_container_width=True):

    if model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    pred = model.predict(input_data)[0]
    risk_label = le_risk.inverse_transform([pred])[0]

    st.success(f"###  Predicted Risk: **{risk_label}**")


    st.markdown("###  Most Influential Features")
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:5]

    for i in idx:
        st.write(f"**{X.columns[i]}** → `{importance[i]:.3f}`")


