# model_baseline_A.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


class TFIDFLogRegBaseline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        """
        texts: list of strings
        labels: list of 0/1 (1 = joke, 0 = non-joke)
        """
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def score(self, text: str) -> float:
        """
        Returns probability that 'text' is a joke.
        """
        X = self.vectorizer.transform([text])
        prob = self.clf.predict_proba(X)[0, 1]
        return float(prob)


def save_model(model: TFIDFLogRegBaseline, path: str):
    joblib.dump({
        "vectorizer": model.vectorizer,
        "clf": model.clf
    }, path)


def load_model(path: str) -> TFIDFLogRegBaseline:
    data = joblib.load(path)
    model = TFIDFLogRegBaseline()
    model.vectorizer = data["vectorizer"]
    model.clf = data["clf"]
    return model
