# -*- coding: utf-8 -*-
"""job posting classification

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ntx17Z3Fs87URC9eRKNYN0XUeRDLtYTC
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        print(f"Scraping page: {page}")
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills
                })
            except Exception as e:
                print(f"Error parsing job block: {e}")
                continue

        time.sleep(1)  # Be nice to the server

    return pd.DataFrame(jobs_list)

# Example use:
if __name__ == "__main__":
    df_jobs = scrape_karkidi_jobs(keyword="data science", pages=2)
    print(df_jobs.head())

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_skills(df):
    # Drop rows with missing skill data
    df = df[df['Skills'].notna()].copy()
    df['Skills_clean'] = df['Skills'].str.lower().str.replace(r'[^a-zA-Z, ]', '', regex=True)
    return df

def custom_tokenizer(text):
    return [token.strip() for token in text.split(',') if token.strip()]

def vectorize_skills(df):
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
    skill_vectors = vectorizer.fit_transform(df['Skills_clean'])
    return skill_vectors, vectorizer

from sklearn.cluster import KMeans
import joblib

def cluster_jobs(skill_vectors, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = model.fit_predict(skill_vectors)
    return cluster_labels, model

def save_model(vectorizer, model, prefix="karkidi"):
    joblib.dump(vectorizer, f"{prefix}_vectorizer.pkl")
    joblib.dump(model, f"{prefix}_kmeans_model.pkl")

def load_model(prefix="karkidi"):
    vectorizer = joblib.load(f"{prefix}_vectorizer.pkl")
    model = joblib.load(f"{prefix}_kmeans_model.pkl")
    return vectorizer, model

def predict_cluster(skills_list, vectorizer, model):
    cleaned = [s.lower().replace(r'[^a-zA-Z, ]', '') for s in skills_list]
    skill_vecs = vectorizer.transform(cleaned)
    return model.predict(skill_vecs)

def notify_user(new_df, user_cluster, user_email=None):
    matched_jobs = new_df[new_df['Cluster'] == user_cluster]
    if matched_jobs.empty:
        print("No new jobs match your interests today.")
    else:
        print("Jobs matching your interests:")
        print(matched_jobs[['Title', 'Company', 'Location']])
        # Optionally send email

if __name__ == "__main__":
    df_jobs = scrape_karkidi_jobs(keyword="data science", pages=2)
    df_jobs = preprocess_skills(df_jobs)
    skill_vectors, vectorizer = vectorize_skills(df_jobs)
    labels, model = cluster_jobs(skill_vectors, n_clusters=5)

    df_jobs['Cluster'] = labels
    save_model(vectorizer, model)

    print(df_jobs.head())

    # Example user cluster preference: 2
    notify_user(df_jobs, user_cluster=2)

!pip install schedule

import schedule
import time
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"

def scrape_page(page, keyword):
    url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    jobs_list = []

    job_blocks = soup.find_all("div", class_="ads-details")
    for job in job_blocks:
        try:
            title = job.find("h4").get_text(strip=True)
            company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
            location = job.find("p").get_text(strip=True)
            experience = job.find("p", class_="emp-exp").get_text(strip=True)
            key_skills_tag = job.find("span", string="Key Skills")
            skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
            summary_tag = job.find("span", string="Summary")
            summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

            jobs_list.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Experience": experience,
                "Summary": summary,
                "Skills": skills
            })
        except Exception:
            continue

    return jobs_list

def scrape_karkidi_jobs_concurrent(keyword="data science", pages=5, max_workers=5):
    jobs_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scrape_page, page, keyword) for page in range(1, pages + 1)]
        for future in as_completed(futures):
            jobs_list.extend(future.result())

    return pd.DataFrame(jobs_list)

def daily_scrape():
    print(f"Starting daily scrape at {datetime.now()}")
    df = scrape_karkidi_jobs_concurrent(keyword="data science", pages=5)
    # Here you can add saving, clustering, etc.
    df.to_csv("daily_jobs.csv", index=False)
    print(f"Scraping finished at {datetime.now()} and saved to daily_jobs.csv")
daily_scrape()
# Schedule scraping every day at 09:00 AM
schedule.every(1).minutes.do(daily_scrape)

print("Scheduler started, waiting for the scheduled time...")

while True:
    schedule.run_pending()
    time.sleep(60)

import joblib

# Assume `model` is your fitted clustering model
# and `vectorizer` is your fitted TfidfVectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

import os
print(os.path.exists("model.pkl"))         # Should return True
print(os.path.exists("vectorizer.pkl"))    # Should return True

import pandas as pd
import joblib

# Load the trained model and vectorizer
model = joblib.load("model.pkl")           # Your trained clustering model
vectorizer = joblib.load("vectorizer.pkl") # Your vectorizer for skills

# Load the new scraped job data
df = pd.read_csv("daily_jobs.csv")

# User's preferred cluster
preferred_cluster = 2  # Change this as needed

# Preprocess and vectorize skills
df['Skills'] = df['Skills'].fillna("").str.lower()
X = vectorizer.transform(df['Skills'])

# Predict clusters
df['Predicted_Cluster'] = model.predict(X)


# Save the complete DataFrame with the Cluster column
df.to_csv("daily_jobs.csv", index=False)

# Filter for user's preferred cluster
matched_jobs = df[df['Predicted_Cluster'] == preferred_cluster]

if not matched_jobs.empty:
    print(f"🔔 ALERT: {len(matched_jobs)} job(s) matched your preferred category!")
    print(matched_jobs[['Title', 'Company', 'Location', 'Skills']])
else:
    print("✅ No new jobs matched your preferred category today.")

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Assume `df` is your DataFrame with a 'Skills' column
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Skills'])

model = KMeans(n_clusters=5, random_state=42)
model.fit(X)
import joblib

# Save trained model and vectorizer to files
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
