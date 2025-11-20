# ==========================
# 🧠 PHISHING EMAIL DETECTOR (LightGBM Version)
# ==========================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

# ========== 1. Load Dataset ==========
print("📂 Loading dataset...")
df = pd.read_csv("phishing_emails.csv")

# Adjust column names (in case they vary)
if 'Email Text' in df.columns:
    df.rename(columns={'Email Text': 'text'}, inplace=True)
if 'Email Type' in df.columns:
    df.rename(columns={'Email Type': 'label'}, inplace=True)

# Drop missing values
df.dropna(subset=['text', 'label'], inplace=True)

# Convert labels to binary (1 = Phishing, 0 = Safe)
df['label'] = df['label'].apply(lambda x: 1 if 'phishing' in x.lower() else 0)

print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
print(df.head())

# ========== 2. Split Data ==========
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ========== 3. Text Vectorization ==========
print("🔤 Converting text to numerical features using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========== 4. Train LightGBM Model ==========
print("⚙️ Training LightGBM model...")
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=-1,
    num_leaves=40,
    random_state=42
)
model.fit(X_train_tfidf, y_train)

# ========== 5. Evaluate ==========
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model trained successfully!")
print(f"✅ Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ========== 6. Save Model & Vectorizer ==========
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n💾 Model and vectorizer saved successfully!")
