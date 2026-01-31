import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="AI Job Risk Prediction", layout="wide")

page = st.sidebar.radio("Navigation", ["Predict", "About"])


SKILL_NAMES = [
    "Creativity",
    "Critical Thinking",
    "Social Intelligence",
    "Manual Dexterity",
    "Routine & Repetitiveness",
    "Technical Complexity",
    "Decision Making",
    "Emotional Intelligence",
    "Automation Exposure",
    "Domain Expertise"
]

JOB_SKILL_PROFILES = {
    "Research Scientist":   [0.75, 0.95, 0.55, 0.15, 0.25, 0.85, 0.85, 0.30, 0.35, 0.95],
    "Construction Worker":  [0.25, 0.45, 0.50, 0.95, 0.70, 0.35, 0.55, 0.45, 0.55, 0.55],
    "Software Engineer":    [0.60, 0.90, 0.45, 0.15, 0.45, 0.90, 0.75, 0.25, 0.55, 0.85],
    "Financial Analyst":    [0.45, 0.90, 0.55, 0.10, 0.60, 0.70, 0.80, 0.35, 0.65, 0.85],
    "AI Engineer":          [0.65, 0.95, 0.40, 0.10, 0.35, 0.95, 0.80, 0.20, 0.55, 0.90],
    "Mechanic":             [0.35, 0.65, 0.45, 0.90, 0.55, 0.55, 0.70, 0.35, 0.45, 0.75],
    "Teacher":              [0.70, 0.80, 0.90, 0.25, 0.35, 0.50, 0.75, 0.85, 0.25, 0.80],
    "HR Specialist":        [0.55, 0.75, 0.90, 0.10, 0.45, 0.45, 0.70, 0.85, 0.35, 0.70],
    "Customer Support":     [0.35, 0.60, 0.85, 0.15, 0.70, 0.35, 0.55, 0.80, 0.65, 0.55],

    "UX Researcher":        [0.80, 0.80, 0.90, 0.30, 0.20, 0.50, 0.70, 0.85, 0.25, 0.75],
    "Lawyer":               [0.60, 0.90, 0.70, 0.20, 0.35, 0.50, 0.80, 0.60, 0.30, 0.90],
    "Data Scientist":       [0.70, 0.90, 0.50, 0.20, 0.30, 0.85, 0.80, 0.30, 0.40, 0.85],
    "Graphic Designer":     [0.95, 0.70, 0.40, 0.40, 0.30, 0.60, 0.60, 0.30, 0.50, 0.70],
    "Retail Worker":        [0.20, 0.30, 0.60, 0.50, 0.90, 0.25, 0.30, 0.55, 0.85, 0.30],
    "Doctor":               [0.50, 0.95, 0.80, 0.60, 0.20, 0.70, 0.95, 0.90, 0.20, 0.95],
    "Truck Driver":         [0.20, 0.40, 0.40, 0.60, 0.80, 0.35, 0.40, 0.30, 0.90, 0.35],
    "Chef":                 [0.70, 0.60, 0.50, 0.85, 0.50, 0.45, 0.60, 0.40, 0.40, 0.70],
    "Nurse":                [0.30, 0.80, 0.80, 0.80, 0.30, 0.50, 0.80, 0.95, 0.30, 0.80],
    "Marketing Manager":    [0.80, 0.70, 0.70, 0.20, 0.40, 0.60, 0.80, 0.60, 0.50, 0.70],
}
def get_default_skills_for_job(job_name):
    job_name = str(job_name).strip()
    return JOB_SKILL_PROFILES.get(job_name, [0.5]*10)

@st.cache_resource
def load_and_train():
    df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

    le_job = LabelEncoder()
    le_edu = LabelEncoder()
    le_risk = LabelEncoder()

    df["Job_Title"] = le_job.fit_transform(df["Job_Title"])
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Risk_Category"] = le_risk.fit_transform(df["Risk_Category"])

    X = df.drop(["Risk_Category", "Automation_Probability_2030"], axis=1)
    y = df["Risk_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest (supports predict_proba)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9
    )
    xgb_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    xgb_pred = np.argmax(xgb_model.predict_proba(X_test), axis=1)

    rf_acc = accuracy_score(y_test, rf_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")
    xgb_f1 = f1_score(y_test, xgb_pred, average="macro")

    metrics = {
        "rf_acc": rf_acc, "rf_f1": rf_f1,
        "xgb_acc": xgb_acc, "xgb_f1": xgb_f1
    }

    return df, X, le_job, le_edu, le_risk, rf_model, xgb_model, metrics


df, X, le_job, le_edu, le_risk, rf_model, xgb_model, metrics = load_and_train()


if page == "About":
    st.title("ℹ About This App")
    st.markdown(
        f"""
### About the Dataset & Project
This app uses a **simulated dataset** modeling how jobs may be impacted by **AI-driven automation by 2030**.

### Models
- Random Forest
- XGBoost

### Model Performance (Test Set)
- **RF Accuracy:** {metrics["rf_acc"]:.3f} | **RF Macro-F1:** {metrics["rf_f1"]:.3f}  
- **XGB Accuracy:** {metrics["xgb_acc"]:.3f} | **XGB Macro-F1:** {metrics["xgb_f1"]:.3f}

### Skills
The app uses **10 skill dimensions** (human/technical/automation-related).  
You can auto-fill skills by job title (recommended) and adjust if needed.

**Created by:** *Saja Hamasha* 🤍
        """
    )
    st.stop()
def set_skills_defaults(skill_values):
    for i, val in enumerate(skill_values, start=1):
        st.session_state[f"skill_{i}"] = float(val)

def get_default_skills_for_job(job_name):
    # If job has profile -> use it, else return neutral defaults
    return JOB_SKILL_PROFILES.get(job_name, [0.5]*10)


st.title(" AI Job Automation Risk Prediction (2030)")
st.write("Predict automation risk level using ML + explain results with SHAP.")
st.markdown("---")

model_choice = st.selectbox("Choose Prediction Model:", ["Random Forest", "XGBoost", "Compare Both"])

st.markdown("##  Job Information")

col1, col2 = st.columns(2)

with col1:
    job_title_encoded = st.selectbox(
        "Job Title:",
        df["Job_Title"].unique(),
        format_func=lambda x: le_job.inverse_transform([x])[0],
        key="job_title_select"
    )

    education_encoded = st.selectbox(
        "Education Level:",
        df["Education_Level"].unique(),
        format_func=lambda x: le_edu.inverse_transform([x])[0],
        key="edu_select"
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

job_title_name = le_job.inverse_transform([job_title_encoded])[0].strip()


left, right = st.columns([2, 1])
with right:
    st.markdown("### Quick Actions")
    if st.button(" Auto-fill skills for this job", use_container_width=True):
        defaults = get_default_skills_for_job(job_title_name)
        set_skills_defaults(defaults)
        st.success("Skills auto-filled ")

    if st.button("↩ Reset skills to 0.5", use_container_width=True):
        set_skills_defaults([0.5]*10)
        st.info("Skills reset ")

if "skill_1" not in st.session_state:
    set_skills_defaults(get_default_skills_for_job(job_title_name))

skills = []
with left:
    for i, skill_name in enumerate(SKILL_NAMES, start=1):
        val = st.slider(
            skill_name,
            0.0, 1.0,
            st.session_state.get(f"skill_{i}", 0.5),
            key=f"skill_{i}",
            help=f"Importance level of {skill_name} for the selected job."
        )
        skills.append(val)

input_data = pd.DataFrame([[
    job_title_encoded, avg_salary, experience, education_encoded,
    ai_exp, tech_growth, *skills
]], columns=X.columns)


def predict_with_model(model, model_name, input_df):
    # Probabilities
    if model_name == "Random Forest":
        proba = model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(proba))
    else:
        proba = model.predict_proba(input_df)[0]  # xgb softprob
        pred_class = int(np.argmax(proba))

    label = le_risk.inverse_transform([pred_class])[0]
    return pred_class, label, proba

def show_probabilities(proba):
    class_names = list(le_risk.classes_)

    labels = [le_risk.inverse_transform([i])[0] for i in range(len(proba))]

    prob_df = pd.DataFrame({"Risk": labels, "Probability": proba})
    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
    st.dataframe(prob_df, use_container_width=True)

def shap_explain_tree(model, input_df, feature_names, pred_class_idx, title_prefix=""):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        sv = shap_values[pred_class_idx][0]
        base = explainer.expected_value[pred_class_idx]
    else:

        if shap_values.ndim == 3:
            sv = shap_values[0, :, pred_class_idx]
            base = explainer.expected_value[pred_class_idx]
        else:
            sv = shap_values[0]
            base = explainer.expected_value

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv
    }).sort_values("shap_value", ascending=False)

    top_pos = shap_df.head(5)
    top_neg = shap_df.tail(5).sort_values("shap_value")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{title_prefix}Top factors pushing risk UP**")
        st.dataframe(top_pos, use_container_width=True)
    with c2:
        st.markdown(f"**{title_prefix}Top factors pushing risk DOWN**")
        st.dataframe(top_neg, use_container_width=True)

    fig, ax = plt.subplots()
    plot_df = pd.concat([top_pos, top_neg]).sort_values("shap_value")
    ax.barh(plot_df["feature"], plot_df["shap_value"])
    ax.set_title(f"{title_prefix}SHAP contributions (predicted class)")
    st.pyplot(fig)

st.markdown("---")

if st.button(" Predict Automation Risk", use_container_width=True):
    results = []

    if model_choice == "Random Forest":
        results.append(("Random Forest", rf_model))
    elif model_choice == "XGBoost":
        results.append(("XGBoost", xgb_model))
    else:
        results.append(("Random Forest", rf_model))
        results.append(("XGBoost", xgb_model))

    for name, model in results:
        pred_class, risk_label, proba = predict_with_model(model, name, input_data)

        st.success(f"### {name} Predicted Risk: **{risk_label}**")

        st.markdown("### Prediction Probabilities")
        show_probabilities(proba)

        st.markdown("### Most Influential Features (Global)")
        importance = model.feature_importances_
        idx = np.argsort(importance)[::-1][:7]
        for i in idx:
            st.write(f"**{X.columns[i]}** → `{importance[i]:.3f}`")

        st.markdown("### SHAP Explanation (This Prediction)")

        try:
            shap_explain_tree(model, input_data, list(X.columns), pred_class_idx=pred_class, title_prefix=f"{name}: ")
        except Exception as e:
            st.warning(f"SHAP explanation failed for {name}: {e}")

        st.markdown("---")
