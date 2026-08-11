# SMS / Email Spam Classifier

A simple end-to-end **NLP + Machine Learning** project that classifies a message as **Spam** or **Not Spam**.

The project covers the complete workflow: text preprocessing, TF-IDF vectorization, model comparison, evaluation, model serialization, and deployment using Streamlit.

---

## Live Demo

**Streamlit App:** _(https://tishya-spam-classifier.streamlit.app/)_

> Enter any SMS/email message and click **Predict** to classify it.

<img width="1033" height="557" alt="image" src="https://github.com/user-attachments/assets/9104998e-044f-42b6-8dd8-c743a6ea4173" />

<!-- Add a screenshot after deployment:
![App Preview](screenshots/app-preview.png)
-->

---

## Project Overview

Spam detection is a **binary text-classification problem**:

- `0` → Not Spam / Ham
- `1` → Spam

The goal is not only to achieve good accuracy, but also to maintain **high precision**, because falsely marking a genuine message as spam can be costly.

---

## ML Pipeline

```text
Raw SMS Data
    ↓
Text Cleaning
    ↓
Tokenization + Stopword Removal + Stemming
    ↓
TF-IDF Vectorization
    ↓
Train / Test Split
    ↓
Model Training & Comparison
    ↓
Evaluation
    ↓
Best Model Selection
    ↓
Pickle Serialization
    ↓
Streamlit Deployment
```

---

## Text Preprocessing

Each message is processed using the same transformation pipeline used during training:

- Convert text to lowercase
- Tokenize the message
- Keep only alphanumeric tokens
- Remove English stopwords
- Remove punctuation
- Apply Porter stemming
- Join tokens back into a cleaned string

Example:

```text
Original:
"Congratulations! You have WON a free prize!!!"

Processed:
"congratul won free prize"
```

---

## Feature Extraction

The cleaned text is converted into numerical features using **TF-IDF**.

```python
TfidfVectorizer(...)
```

The fitted vectorizer is saved as:

```text
vectorizer.pkl
```

This ensures that new user messages are transformed using the **same vocabulary and feature mapping** that were used during training.

---

## Models Compared

The following classifiers were evaluated:

- Support Vector Classifier
- K-Nearest Neighbors
- Multinomial Naive Bayes
- Decision Tree
- Logistic Regression
- Random Forest
- AdaBoost
- Bagging Classifier
- Extra Trees
- Gradient Boosting
- XGBoost

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Final Model

### Multinomial Naive Bayes

**Multinomial Naive Bayes** was selected as the final classifier because it performed strongly on the TF-IDF text features and maintained very high precision.

Why it fits this problem well:

- Designed for frequency-based text features
- Efficient on high-dimensional sparse data
- Fast to train and predict
- Strong baseline for NLP classification
- Performs well without unnecessary model complexity

> **Key takeaway:** A more complex model is not always a better model.  
> In this project, simpler Naive Bayes performed better than several ensemble approaches.

---

## Experiments Tried

Several techniques were explored to improve model performance:

- TF-IDF `max_features`
- Feature scaling
- Additional numerical features
- Multiple ML classifiers
- Voting Classifier
- Stacking
- Ensemble models

These experiments helped confirm that Multinomial Naive Bayes was still the most suitable model for this dataset.

---

## Streamlit Prediction Flow

```text
User enters message
        ↓
transform_text()
        ↓
tfidf.transform()
        ↓
model.predict()
        ↓
Spam / Not Spam
```

The Streamlit app loads:

```text
vectorizer.pkl  → fitted TF-IDF vectorizer
model.pkl       → trained classifier
```

---

## Tech Stack

**Language**
- Python

**Machine Learning / NLP**
- Scikit-learn
- NLTK
- XGBoost

**Data Analysis**
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Seaborn
- WordCloud

**Deployment**
- Streamlit
- Streamlit Community Cloud

**Version Control**
- Git
- GitHub

---

## Project Structure

```text
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
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Tishya22/spam_classifier.git
cd spam_classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the app

```bash
python -m streamlit run app.py
```

---

## What I Learned

This project helped me practice:

- NLP text preprocessing
- Tokenization and stopword removal
- Stemming
- TF-IDF vectorization
- Binary classification
- Model evaluation
- Precision vs. accuracy for imbalanced datasets
- Comparing multiple classifiers
- Ensemble learning
- Pickle model serialization
- Building a Streamlit interface
- Git and GitHub workflow
- Deploying an ML application

---

## Future Improvements

- Tune TF-IDF `ngram_range`, `min_df`, and `max_df`
- Tune MultinomialNB `alpha`
- Try Complement Naive Bayes
- Experiment with character-level TF-IDF
- Add engineered features such as URL count, digit count, and punctuation count
- Add prediction confidence to the UI
- Perform deeper error analysis on false positives and false negatives
- Compare with transformer-based text classifiers

---

## Author

**Tishya Misra**  
B.Tech Computer Science & Engineering

GitHub: [Tishya22](https://github.com/Tishya22)

---

If you found this project useful, consider starring the repository.
