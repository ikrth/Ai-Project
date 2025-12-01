from model_baseline_A import load_model


def generate_for_word_pair(model, word1: str, word2: str) -> str:
    """
    Generate a joke that uses both word1 and word2.
    We create several templates, score them with the TF-IDF + LogReg model,
    and return the highest-scoring one.
    """
    candidates = [
        f"Why did the {word1} meet the {word2}? Because reality was not weird enough already.",
        f"The {word1} walked into a {word2} and everyone pretended it was normal.",
        f"When your {word1} starts giving advice to your {word2}, it is time to log off the internet.",
        f"My {word1} and my {word2} started a podcast. It is just them arguing about whose fault it is.",
        f"I tried to use a {word1} as a {word2}. Now the manual just says 'do not be this person.'",
        f"If you mix a {word1} with a {word2}, you either get genius innovation or a very weird Tuesday.",
        f"The support group for people who confused a {word1} with a {word2} meets every Thursday.",
    ]

    scores = [model.score(c) for c in candidates]
    best_idx = scores.index(max(scores))
    return candidates[best_idx]


def generate_for_headline(model, headline: str) -> str:
    """
    Generate a joke related to a news-style headline.
    """
    candidates = [
        f"{headline} Honestly, that already sounds like the setup to a bad joke.",
        f"{headline} Somewhere, a comedian just crossed this off their set list.",
        f"{headline} I guess this is what happens when we leave humans unsupervised.",
        f"{headline} And here I was thinking my life choices were questionable.",
        f"{headline} At this point, the writers of reality are clearly out of ideas.",
        f"{headline} Meanwhile, the universe quietly whispers: 'Hold my coffee.'",
    ]

    scores = [model.score(c) for c in candidates]
    best_idx = scores.index(max(scores))
    return candidates[best_idx]


def interactive_loop(model):
    """
    Let the user generate jokes repeatedly until they choose to quit.
    """
    while True:
        print("\n=== Humor Generator (Baseline A) ===")
        print("1) Generate joke from word pair")
        print("2) Generate joke from headline")
        print("q) Quit")
        choice = input("Choose mode (1/2/q): ").strip().lower()

        if choice in ("q", "quit", "3"):
            print("Goodbye! Thanks for testing the humor generator.")
            break

        elif choice == "1":
            w1 = input("Enter word1: ").strip()
            w2 = input("Enter word2: ").strip()
            joke = generate_for_word_pair(model, w1, w2)
            print("\nGenerated joke:")
            print(joke)

        elif choice == "2":
            h = input("Enter headline: ").strip()
            joke = generate_for_headline(model, h)
            print("\nGenerated joke:")
            print(joke)

        else:
            print("Invalid choice. Please enter 1, 2, or q.")


def main():
    model = load_model("baseline_A_tfidf_logreg.joblib")
    interactive_loop(model)


if __name__ == "__main__":
    main()
