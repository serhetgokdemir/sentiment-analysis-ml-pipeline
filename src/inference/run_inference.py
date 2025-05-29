import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "src"))

from data_loader import load_data
import joblib
import pandas as pd

def main():
    model_path = os.path.join(project_root, "outputs", "models", "final_model.pkl")
    vectorizer_path = os.path.join(project_root, "outputs", "models", "tfidf_vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    test_data = load_data("test.csv")
    X_test = test_data["review"]

    X_test_tfidf = vectorizer.transform(X_test)

    test_preds = model.predict(X_test_tfidf)

    predictions_dir = os.path.join(project_root, "outputs", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_path = os.path.join(predictions_dir, "test_predictions.csv")
    pd.DataFrame({"review": X_test, "predicted_sentiment": test_preds}).to_csv(predictions_path, index=False)

    print(f"Predictions saved at: {predictions_path}")

if __name__ == "__main__":
    main()