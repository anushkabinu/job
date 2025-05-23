import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load scraped job data
df = pd.read_csv("daily_jobs.csv")
df['Skills'] = df['Skills'].fillna("").str.lower()

# Vectorize and predict clusters
X = vectorizer.transform(df['Skills'])
df['Predicted_Cluster'] = model.predict(X)

# Streamlit UI
st.title("🔍 Job Alert System")
st.markdown("Get job listings that match your interests.")

# User selects cluster
unique_clusters = sorted(df['Predicted_Cluster'].unique())
selected_cluster = st.selectbox("Choose your preferred job category (Cluster):", unique_clusters)

# Show results
matched = df[df['Predicted_Cluster'] == selected_cluster]
st.write(f"### 🎯 Found {len(matched)} jobs in your preferred category.")

if not matched.empty:
    st.dataframe(matched[['Title', 'Company', 'Location', 'Skills']])
else:
    st.info("No matching jobs at the moment.")

# Optional: Download CSV
st.download_button("Download Matching Jobs as CSV", matched.to_csv(index=False), file_name="matching_jobs.csv")
