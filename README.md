# Emotion Classification and Training Scripts

## Overview

This repository contains two Python scripts designed to handle different aspects of emotion classification:

1. **`app.py`** - A Streamlit-based web application for real-time emotion classification of user-provided text.  
2. **`train_bert_model.py`** - A script for training and fine-tuning a BERT-based model for emotion classification.

---

## Scripts

### 1. `app.py`

This script provides a user-friendly web interface for emotion analysis of textual data. It utilizes a pre-trained text classification model and logs results to a database for auditing and validation.

#### Features:
- **Real-time Emotion Classification**: Predicts emotions from user-input text.
- **Database Logging**: Logs predictions, latency, and drift metrics to a PostgreSQL database.
- **Evaluation Metrics**: Calculates and displays accuracy, precision, and recall from validation data.

#### Requirements:
- **Streamlit** for the web application.
- **Hugging Face Transformers** for the model pipeline.
- **PostgreSQL** for database logging.
- **Environment Variables**: Ensure `.env` contains `DB_HOST`, `DB_USER`, and `DB_PASSWORD`.
### 2. `train_bert_model.py`

This script trains a BERT-based model for emotion classification using a labeled dataset. It fine-tunes the `cardiffnlp/twitter-roberta-base-sentiment-latest` model for six emotion categories: **Sadness**, **Joy**, **Love**, **Anger**, **Fear**, and **Surprise**.

#### Features:
- **Exploratory Data Analysis (EDA)**: Visualizes text length and label distributions.
- **Data Preprocessing**: Tokenizes text for model input.
- **Model Fine-Tuning**: Trains the model with specified hyperparameters.
- **Evaluation**: Outputs accuracy and F1 score on the test dataset.

#### Requirements:
- **Hugging Face Transformers** for model and tokenizer.
- **Matplotlib** for data visualization.
- **Datasets** library for data loading and processing.


**Front-End**: http://162.243.25.49:8501/
