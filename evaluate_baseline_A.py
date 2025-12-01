import pandas as pd
from model_baseline_A import load_model
from generate_baseline_A import generate_for_word_pair, generate_for_headline


def main():
    # 1. Load the trained TF-IDF + LogReg model
    model = load_model("baseline_A_tfidf_logreg.joblib")

    # 2. Load the MWAHAHA Subtask A English data
    df = pd.read_csv(
        r"mwahaha_dev_tasks-ab_v2\task-a-en.tsv",  # adjust folder name if different
        sep="\t"
    )

    generated_texts = []
    constraint_types = []
    word1_list = []
    word2_list = []
    headline_list = []

    # 3. Generate jokes for each row
    for _, row in df.iterrows():
        word1 = str(row["word1"])
        word2 = str(row["word2"])
        headline = str(row["headline"])

        if headline != "-" and headline.strip():
            constraint_type = "headline"
            joke = generate_for_headline(model, headline)
        else:
            constraint_type = "word"
            joke = generate_for_word_pair(model, word1, word2)

        generated_texts.append(joke)
        constraint_types.append(constraint_type)
        word1_list.append(word1)
        word2_list.append(word2)
        headline_list.append(headline)

    out_df = pd.DataFrame({
        "constraint_type": constraint_types,
        "word1": word1_list,
        "word2": word2_list,
        "headline": headline_list,
        "generated_joke": generated_texts,
    })
    out_df.to_csv("results_baseline_A_task_a_en.tsv", sep="\t", index=False)
    print("Saved generated jokes to results_baseline_A_task_a_en.tsv")

    # 4. Simple evaluation metrics

    total_word_rows = 0
    satisfied_word_rows = 0

    total_headline_rows = 0
    satisfied_headline_rows = 0

    scores = []

    for i in range(len(out_df)):
        constraint_type = out_df.loc[i, "constraint_type"]
        joke = str(out_df.loc[i, "generated_joke"])

        if constraint_type == "word":
            total_word_rows += 1
            w1 = str(out_df.loc[i, "word1"]).lower()
            w2 = str(out_df.loc[i, "word2"]).lower()
            joke_lower = joke.lower()
            if w1 in joke_lower and w2 in joke_lower:
                satisfied_word_rows += 1
        else:
            total_headline_rows += 1
            headline = str(out_df.loc[i, "headline"]).lower()
            joke_lower = joke.lower()
            headline_words = [w for w in headline.split() if len(w) > 3]
            if any(w in joke_lower for w in headline_words):
                satisfied_headline_rows += 1

        scores.append(model.score(joke))

    word_constraint_rate = (satisfied_word_rows / total_word_rows) if total_word_rows > 0 else 0.0
    headline_overlap_rate = (satisfied_headline_rows / total_headline_rows) if total_headline_rows > 0 else 0.0
    avg_humor_score = sum(scores) / len(scores)

    print("\n=== Baseline A (TF-IDF + LogReg + templates) metrics ===")
    print(f"Word-pair constraint satisfaction: {word_constraint_rate:.3f}")
    print(f"Headline overlap rate:            {headline_overlap_rate:.3f}")
    print(f"Average humor score:              {avg_humor_score:.3f}")


if __name__ == "__main__":
    main()
