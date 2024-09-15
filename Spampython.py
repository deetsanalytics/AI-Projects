import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load the dataset
df = pd.read_csv("spam.csv", encoding="latin1")

# Preprocess the data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["text"] = df["v2"].apply(preprocess_text)

# Split the data into training and testing sets
X = df["v2"]
y = df["v1"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'alpha': [0.1, 0.5, 1, 5, 10],
    'fit_prior': [True, False]
}

# Perform grid search CV
model = MultinomialNB()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_

# Create a Streamlit webapp
st.title("Spam Detection Webapp")

input_text = st.text_input("Enter a text message:")

if st.button("Predict"):
    input_text_tfidf = vectorizer.transform([input_text])
    prediction = best_model.predict(input_text_tfidf)[0]
    if prediction == 0:
        st.write("This message is not spam.")
    else:
        st.write("This message is spam.")