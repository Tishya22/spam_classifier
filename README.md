📩 SMS / Email Spam Classifier

A lightweight NLP-based machine learning application that classifies text messages as Spam or Not Spam using TF-IDF vectorization and a Multinomial Naive Bayes classifier.

The project covers the complete ML workflow — from data cleaning and exploratory analysis to feature extraction, model comparison, evaluation, serialization, and deployment with Streamlit.

🚀 Live Demo

Streamlit App: Add your deployed Streamlit URL here

Enter any SMS/email message and the application will instantly classify it as Spam or Not Spam.

✨ Project Highlights

Built an end-to-end NLP text classification pipeline

Cleaned and transformed raw SMS text using NLTK

Applied TF-IDF to convert text into numerical features

Compared multiple machine learning classifiers

Evaluated models using Accuracy, Precision, Recall, F1-score, and Confusion Matrix

Selected Multinomial Naive Bayes as the final model based on its strong performance and high precision

Saved the trained model and TF-IDF vectorizer using Pickle

Built an interactive user interface using Streamlit

Structured the application for deployment on Streamlit Community Cloud

🧠 Problem Statement

Spam filtering is a binary text-classification problem:

0 → Ham / Not Spam

1 → Spam

The objective is not only to achieve good overall accuracy, but also to maintain high precision, because incorrectly classifying a genuine message as spam can be costly.

🔄 Machine Learning Workflow

Raw SMS Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Train / Test Split
      ↓
Model Training & Comparison
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Pickle Serialization
      ↓
Streamlit Web App
      ↓
Deployment

🧹 Text Preprocessing

Each incoming message is processed using the same transformation pipeline used during training:

Convert text to lowercase

Tokenize the message

Remove non-alphanumeric tokens

Remove English stopwords

Remove punctuation

Apply Porter Stemming

Join the processed tokens back into a string

Example:

Original:
"Congratulations! You have WON a free prize!!!"

Processed:
"congratul won free prize"

🔢 Feature Extraction — TF-IDF

The cleaned text is converted into numerical features using:

TfidfVectorizer(...)

TF-IDF gives higher importance to words that are useful for distinguishing one message from another while reducing the importance of very common words.

The trained TF-IDF vectorizer is stored in:

vectorizer.pkl

This ensures that new messages are transformed using the same vocabulary and feature mapping used during model training.

🤖 Models Compared

The project experiments with several machine learning classifiers:

Support Vector Classifier

K-Nearest Neighbors

Multinomial Naive Bayes

Decision Tree

Logistic Regression

Random Forest

AdaBoost

Bagging Classifier

Extra Trees

Gradient Boosting

XGBoost

The models were compared using multiple evaluation metrics rather than relying on accuracy alone.

🏆 Final Model — Multinomial Naive Bayes

Multinomial Naive Bayes was selected as the final classifier.

Why it worked well:

Well suited for text classification

Performs efficiently with high-dimensional sparse features

Works naturally with TF-IDF / word-frequency based representations

Produced very strong precision, which is particularly important for spam detection

Remained competitive even when compared with more complex ensemble models

This project also reinforced an important ML principle:

A more complex model is not always a better model.

📊 Evaluation Metrics

The models were evaluated using:

Accuracy

Measures the percentage of all predictions that were correct.

Precision

Measures how reliable the model's Spam predictions are.

Precision = TP / (TP + FP)

High precision is especially important here because a False Positive means a genuine message is incorrectly classified as Spam.

Recall

Measures how much of the actual Spam was successfully detected.

Recall = TP / (TP + FN)

F1-Score

Balances Precision and Recall into one metric.

Confusion Matrix

Used to inspect:

True Positives

True Negatives

False Positives

False Negatives

🌐 Streamlit Application

The deployed application follows this prediction pipeline:

User enters a message
        ↓
transform_text()
        ↓
TF-IDF transform()
        ↓
Loaded trained model
        ↓
model.predict()
        ↓
Spam / Not Spam

The application loads:

vectorizer.pkl → trained TF-IDF vectorizer
model.pkl      → trained Multinomial Naive Bayes classifier

model.predict() then uses the patterns learned during training to classify the new TF-IDF vector.

🛠️ Tech Stack

Language

Python

Machine Learning / NLP

Scikit-learn

NLTK

XGBoost

Data Analysis

Pandas

NumPy

Visualization

Matplotlib

Seaborn

WordCloud

Deployment

Streamlit

Streamlit Community Cloud

Version Control

Git

GitHub

📂 Project Structure

spam_classifier/
│
├── app.py
├── sms_spam_detection.ipynb
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md

File Description

File

Purpose

app.py

Streamlit application

sms_spam_detection.ipynb

Data analysis, preprocessing, model training and evaluation

spam.csv

Dataset used for training

model.pkl

Serialized trained classifier

vectorizer.pkl

Serialized TF-IDF vectorizer

requirements.txt

Python dependencies

.gitignore

Files excluded from version control

⚙️ Run Locally

1. Clone the repository

git clone https://github.com/Tishya22/spam_classifier.git
cd spam_classifier

2. Create a virtual environment

python -m venv .venv

3. Activate the environment

Windows PowerShell

.\.venv\Scripts\Activate.ps1

4. Install dependencies

python -m pip install -r requirements.txt

5. Run the application

python -m streamlit run app.py

The application will open in your browser.

📌 Key Learning Outcomes

Through this project, I worked with:

Text preprocessing and tokenization

Stopword removal

Stemming

Exploratory Data Analysis for NLP

TF-IDF vectorization

Sparse feature matrices

Binary classification

Model comparison

Precision vs Accuracy for imbalanced datasets

Confusion matrices

Ensemble learning

Hyperparameter experimentation

Model serialization using Pickle

Building a Streamlit interface

Virtual environments

Git / GitHub workflow

ML application deployment

🔬 Experiments & Optimization

Several techniques were explored while trying to improve model performance, including:

Limiting TF-IDF vocabulary using max_features

Comparing multiple classifiers

Adding engineered numerical features

Feature scaling

Ensemble methods

Voting classifiers

Stacking classifiers

Despite experimenting with more complex approaches, Multinomial Naive Bayes remained the strongest choice for this dataset.

This demonstrates the importance of selecting a model based on measured performance and suitability for the problem, rather than model complexity.

🔮 Future Improvements

Potential improvements include:

Tune TF-IDF ngram_range, min_df, and max_df

Hyperparameter tuning for Multinomial Naive Bayes

Experiment with Complement Naive Bayes

Try character-level TF-IDF features

Add spam-specific features such as URL count, digit count and punctuation frequency

Use cross-validation for more robust model evaluation

Perform systematic error analysis on False Positives and False Negatives

Experiment with transformer-based models such as BERT / DistilBERT

Add prediction confidence to the Streamlit UI

Improve the deployed interface with examples and visual feedback

👩‍💻 Author

Tishya MisraB.Tech Computer Science & Engineering

GitHub: Tishya22git 