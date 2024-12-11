import streamlit as st
from transformers import pipeline
import psycopg2
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to the database
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database="defaultdb",
        user=os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        port=25060,
        sslmode="require"
    )
    return conn

# Function to log predictions
def log_to_database(input_text, prediction, latency):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Calculate distribution metrics (e.g., mean of probabilities)
    probabilities = [item['score'] for item in prediction]
    mean_prob = np.mean(probabilities)

    # Insert the log into the database
    cursor.execute("""
        INSERT INTO audit_logs (input_text, prediction, latency, drift_metric)
        VALUES (%s, %s, %s, %s)
    """, (input_text, json.dumps(prediction),latency, mean_prob))
    conn.commit()
    cursor.close()
    conn.close()

# Function to log validation data for evaluation
def log_validation_data(true_label, predicted_label):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO validation_logs (true_label, predicted_label)
        VALUES (%s, %s)
    """, (true_label, predicted_label))
    conn.commit()
    cursor.close()
    conn.close()

# Fetch validation data for metric calculation
def fetch_validation_data():
    conn = get_db_connection()
    query = "SELECT true_label, predicted_label FROM validation_logs"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Load the model and tokenizer
classifier = pipeline("text-classification", model="/root/saved_model", tokenizer="/roo>

# Label mapping
label_mapping = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

# Streamlit App
st.title("Emotion Classification App")
st.write("Analyze customer feedback for emotions.")

# Input Text Box
input_text = st.text_area("Enter Text:", placeholder="Type something...")

# Button for Prediction
if st.button("Classify"):
    if input_text.strip():
        results = classifier(input_text)
        start_time = time.time()
        latency = time.time() - start_time  # Measure time taken for prediction

        # Extract the label and map it
        label_number = int(results[0]['label'].split('_')[1])  # Extract the label numb>
        emotion = label_mapping[label_number]
        score = round(results[0]['score'], 2)  # Round score to 2 decimal places

        # Simulated true label (replace this with actual ground truth if available)
        true_label = 1  # Example true label
        predicted_label = label_number

        # Log predictions and validation data
        latency = time.time() - time.time()
        log_to_database(input_text, results, latency)
        log_validation_data(true_label, predicted_label)

        # Display only emotion and score
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Score:** {score}")
    else:
        st.warning("Please enter some text!")

# Display Model Evaluation Metrics
st.write("### Model Evaluation Metrics")

