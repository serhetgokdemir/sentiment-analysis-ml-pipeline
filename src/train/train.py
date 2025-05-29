import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "src"))

from data_loader import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    train_data = load_data("train.csv")
    X = train_data["review"]
    y = train_data["sentiment"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    val_preds = model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")

    model_dir = os.path.join(project_root, "outputs", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "final_model.pkl")
    joblib.dump(model, model_path)

    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)

if __name__ == "__main__":
    main()