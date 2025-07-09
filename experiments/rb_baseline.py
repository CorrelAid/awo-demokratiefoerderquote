from aim import Run
import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import nltk
from aim import Text
from nltk.stem.snowball import GermanStemmer


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class RBClassifier:
    def __init__(
        self,
        min_freq=2,
        max_features=100,
        score_threshold=0.85,
        use_stemming=True,
        ngram_range=(1, 2),
    ):
        self.min_freq = min_freq
        self.max_features = max_features
        self.score_threshold = score_threshold
        self.use_stemming = use_stemming
        self.ngram_range = ngram_range
        self.rules = []
        self.is_fitted = False

        if use_stemming:
            self.stemmer = GermanStemmer()
        else:
            self.stemmer = None

    def _preprocess_text(self, text):
        """Preprocess text with optional stemming"""
        if not isinstance(text, str):
            return ""

        text = text.lower()

        if self.use_stemming:
            tokens = word_tokenize(text)
            stemmed_tokens = [
                self.stemmer.stem(token) for token in tokens if token.isalnum()
            ]
            return " ".join(stemmed_tokens)

        return text

    def _extract_words(self, texts):
        """Extract word frequencies from texts"""
        if not texts:
            return {}

        vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_freq,
            max_features=self.max_features,
            token_pattern=r"\b\w+\b",
        )

        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # sum frequencies across all documents
        word_freq = X.sum(axis=0).A1
        return dict(zip(feature_names, word_freq))

    def _find_discriminative_patterns(self, pos_counter, neg_counter):
        """Find discriminative patterns based on frequency scores"""
        rules = []

        for word, pos_freq in pos_counter.items():
            neg_freq = neg_counter.get(word, 0)

            if pos_freq > 0:
                # what percentage of the time this pattern appears in positive samples versus all samples?
                score = pos_freq / (pos_freq + neg_freq + 1)
                if score > self.score_threshold:
                    rules.append((word, score))

        return sorted(rules, key=lambda x: x[1], reverse=True)

    def fit(self, texts, labels):
        processed_texts = [self._preprocess_text(text) for text in texts]

        positive_texts = [
            processed_texts[i] for i, label in enumerate(labels) if label == 1
        ]
        negative_texts = [
            processed_texts[i] for i, label in enumerate(labels) if label == 0
        ]

        # Extract word frequencies
        pos_counter = self._extract_words(positive_texts)
        neg_counter = self._extract_words(negative_texts)

        # Find discriminative patterns
        self.rules = self._find_discriminative_patterns(pos_counter, neg_counter)

        self.is_fitted = True
        return self

    def predict(self, texts):
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        if isinstance(texts, str):
            texts = [texts]

        if not self.rules:
            return [0] * len(texts)

        predictions = []
        for text in texts:
            # Preprocess the text for prediction
            processed_text = self._preprocess_text(text)

            # Check if any rule matches
            prediction = 0
            for pattern, weight in self.rules:
                if pattern in processed_text:
                    prediction = 1
                    break  # first match determines positive classification
            predictions.append(prediction)

        return predictions

    def get_learned_rules(self):
        """Get the learned rules"""
        return self.rules if self.is_fitted else []


def rb_baseline():
    df = pl.read_csv("data/labeled/19_06/to_classify.csv")

    texts = df["description_title_cats_compact"].to_list()
    true_labels = df["label"].to_list()

    print("Connecting to aim repo..\n")
    aim_run = Run(
        repo="aim://homelab-hetzner-0.tail860809.ts.net:53800",
        experiment="dhh_rb_baseline_full_data",
    )
    print("Connected ✔️\n")

    aim_run.add_tag("rb_baseline_full_data")

    score_threshold = 0.94
    max_features = 1000
    min_freq = 3
    use_stemming = True
    ngram_range = (1, 2)

    aim_run["hparams"] = {
        "score_threshold": score_threshold,
        "max_features": max_features,
        "min_freq": min_freq,
        "use_stemming": use_stemming,
        "ngram_range": ngram_range,
    }

    print(f"Total dataset: {len(texts)} samples")
    print(f"Positive samples: {sum(true_labels)}")
    print(f"Negative samples: {len(true_labels) - sum(true_labels)}")
    print("Training and evaluating on full dataset")

    classifier = RBClassifier(
        min_freq=min_freq,
        max_features=max_features,
        score_threshold=score_threshold,
        use_stemming=use_stemming,
        ngram_range=ngram_range,
    )
    classifier.fit(texts, true_labels)

    # evaluate on all data
    predictions = classifier.predict(texts)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    aim_run.track(accuracy, name="accuracy", context={"subset": "full_data"})
    aim_run.track(precision, name="precision", context={"subset": "full_data"})
    aim_run.track(recall, name="recall", context={"subset": "full_data"})
    aim_run.track(f1, name="f1_score", context={"subset": "full_data"})

    num_rules = len(classifier.get_learned_rules())

    print(f"\n{'='*60}")
    print("PERFORMANCE ON FULL DATASET")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Score':<12}")
    print(f"{'-'*30}")
    print(f"{'Accuracy':<12} {accuracy:<12.2%}")
    print(f"{'Precision':<12} {precision:<12.2%}")
    print(f"{'Recall':<12} {recall:<12.2%}")
    print(f"{'F1-Score':<12} {f1:<12.2%}")
    print(f"{'Rules':<12} {num_rules:<12}")

    print(f"\n{'='*60}")
    print(
        "TOP 10 LEARNED RULES (What percentage of the time this pattern appears in positive samples versus ALL samples (positive + negative)"
    )
    print(f"{'='*60}")

    top_rules = classifier.get_learned_rules()[:10]
    print(f"{'Rank':<6} {'Pattern':<30} {'Score':<10}")
    print(f"{'-'*50}")

    for i, (pattern, score) in enumerate(top_rules, 1):
        print(f"{i:<6} {pattern:<30} {score:<10.4f}")

    rules_text = "TOP 10 LEARNED RULES:\n"
    rules_text += f"{'Rank':<6} {'Pattern':<30} {'Score':<10}\n"
    rules_text += f"{'-'*50}\n"

    for i, (pattern, score) in enumerate(top_rules, 1):
        rules_text += f"{i:<6} {pattern:<30} {score:<10.4f}\n"

    aim_text = Text(rules_text)
    aim_run.track(aim_text, name="top_rules", step=0)

    aim_run.close()


if __name__ == "__main__":
    rb_baseline()
