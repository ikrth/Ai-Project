from model_baseline_A import TFIDFLogRegBaseline, save_model

def main():
    # Tiny toy dataset for now (you can improve this later)
    joke_texts = [
        "Why did the chicken cross the road? To get to the other side.",
        "I told my computer a joke but it did not laugh, it just crashed.",
        "My boss told me to have a good day, so I went home."
    ]
    non_joke_texts = [
        "The stock market closed higher today.",
        "The weather is expected to be sunny.",
        "This is a user manual for a washing machine."
    ]

    texts = joke_texts + non_joke_texts
    labels = [1] * len(joke_texts) + [0] * len(non_joke_texts)

    model = TFIDFLogRegBaseline()
    model.fit(texts, labels)

    save_model(model, "baseline_A_tfidf_logreg.joblib")
    print("Model trained and saved to baseline_A_tfidf_logreg.joblib")

if __name__ == "__main__":
    main()
