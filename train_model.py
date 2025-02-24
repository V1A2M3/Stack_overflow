# Re-import necessary libraries
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Reload dataset
file_path = "C:/Users/chitt/Downloads/bids.csv"
df = pd.read_csv(file_path)

# Ensure the 'merchandise' column exists
if "merchandise" not in df.columns:
    raise ValueError("The dataset does not contain a 'merchandise' column.")

# Handle missing values
df["merchandise"] = df["merchandise"].astype(str).fillna("")

# Create labels (Multi-Label Encoding)
df["tags"] = df["merchandise"].apply(lambda x: [x.lower()] if pd.notna(x) else [])

# Convert tags to multi-label binary format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["tags"])

# TF-IDF Vectorization + Model Training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["merchandise"])

# Use OneVsRestClassifier with LogisticRegression
model = OneVsRestClassifier(LogisticRegression(max_iter=500))
model.fit(X, y)

# Save Model & Vectorizer
model_path = "/mnt/data/tagging_model.pkl"
vectorizer_path = "/mnt/data/tfidf_vectorizer.pkl"
mlb_path = "/mnt/data/mlb.pkl"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(mlb, mlb_path)

model_path, vectorizer_path, mlb_path
