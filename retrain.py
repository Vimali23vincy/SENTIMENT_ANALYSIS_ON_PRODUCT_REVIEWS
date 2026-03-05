import pandas as pd
import joblib
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os

# Download NLTK data
nltk.download('stopwords')

# Negations list
NEGATIONS = {"no", "not", "nor", "neither", "never", "none", "nt", "isnt", "wasnt", "arent", "werent", "dont", "doesnt", "didnt", "hasnt", "havent", "hadnt", "shouldnt", "wouldnt", "couldnt", "mightnt", "mustnt"}
stop_words = set(stopwords.words('english')) - NEGATIONS

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("n't", " nt")
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("Starting training on Optimized Dataset...")

# Load the NEW optimized dataset
df = pd.read_csv("Datasets_Merged.csv")
df['clean_text'] = df['review'].apply(clean_text)

# We use a Pipeline to bundle the vectorizer and classifier
# LinearSVC is generally stronger than Random Forest for text classification
# CalibratedClassifierCV is used to get probability estimates from SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2), 
        max_features=10000, 
        min_df=2, 
        sublinear_tf=True
    )),
    ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, C=1.0)))
])

# Train
print("Fitting the Model...")
pipeline.fit(df['clean_text'], df['sentiment'])

# Evaluate (Self-Check)
y_pred = pipeline.predict(df['clean_text'])
print("\nTraining Metrics:")
print(classification_report(df['sentiment'], y_pred))

# Save the entire pipeline
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(pipeline, "models/sentiment_pipeline.pkl")

print("Training Complete! Optimized SVM Pipeline saved to models/sentiment_pipeline.pkl")
