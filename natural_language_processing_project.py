import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    def tokenize_text(self, text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return tokens

    def stem_and_lemmatize(self, tokens):
        stemmed = [self.stemmer.stem(word) for word in tokens]
        lemmatized = [self.lemmatizer.lemmatize(word) for word in stemmed]
        return lemmatized

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        return self.stem_and_lemmatize(tokens)

class NLPModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def fit(self, X_train, y_train):
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)

    def predict(self, X_test):
        X_test_vectorized = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vectorized)

    def evaluate(self, y_test, y_pred):
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['text'], data['label']

def main():
    # Load data
    file_path = 'data.csv'  # Make sure to have a data.csv file in the directory
    X, y = load_data(file_path)

    # Preprocess data
    preprocessor = TextPreprocessor()
    X_processed = [' '.join(preprocessor.preprocess(text)) for text in X]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train model
    model = NLPModel()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate model
    model.evaluate(y_test, y_pred)

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.countplot(y)
    plt.title('Class Distribution')
    plt.xlabel('Class Labels')
    plt.ylabel('Counts')
    plt.show()

if __name__ == "__main__":
    main()