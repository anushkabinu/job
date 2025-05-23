import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Job Notifier", layout="wide")
st.title("💼 Job Notifier Based on Your Skills")

# Check required files
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.error("❌ Model or vectorizer files are missing! Please upload 'model.pkl' and 'vectorizer.pkl'.")
    st.stop()

if not os.path.exists("daily_jobs.csv"):
    st.error("❌ Job data file 'daily_jobs.csv' is missing! Please run the scraper first.")
    st.stop()

# Load resources
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("daily_jobs.csv")

# Check if 'Predicted_Cluster' column exists, else create it
if 'Predicted_Cluster' not in df.columns:
    # Fill missing skills with empty string and lower case for consistency
    df['Skills'] = df['Skills'].fillna("").str.lower()
    X = vectorizer.transform(df['Skills'])
    df['Predicted_Cluster'] = model.predict(X)
    # Optional: save updated CSV with clusters to avoid recomputing next time
    df.to_csv("daily_jobs.csv", index=False)

# Display job data
st.subheader("📋 Latest Job Listings")
st.dataframe(df[['Title', 'Company', 'Location', 'Skills']])

# User input
st.subheader("🔍 Get Notified for Jobs Matching Your Skills")
user_skills_input = st.text_input("Enter your skills (comma-separated)", "")

if user_skills_input:
    user_skills_list = [s.strip() for s in user_skills_input.split(",") if s.strip()]
    user_skills_str = " ".join(user_skills_list).lower()
    user_vector = vectorizer.transform([user_skills_str])
    user_cluster = model.predict(user_vector)[0]

    st.success(f"✅ Based on your skills, you're in cluster #{user_cluster}.")

    matched_jobs = df[df['Predicted_Cluster'] == user_cluster]

    if not matched_jobs.empty:
        st.subheader("📬 Jobs That Match Your Skills")
        st.dataframe(matched_jobs[['Title', 'Company', 'Location', 'Skills']])
    else:
        st.info("No jobs found in your preferred cluster today. Please check back later.")
