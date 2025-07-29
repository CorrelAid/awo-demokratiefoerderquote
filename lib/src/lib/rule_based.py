from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.snowball import GermanStemmer


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
        rules = []

        for word, pos_freq in pos_counter.items():
            neg_freq = neg_counter.get(word, 0)

            if pos_freq > 0:
                # measure of a tokens occurrence in positive samples versus all samples
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

        pos_counter = self._extract_words(positive_texts)
        neg_counter = self._extract_words(negative_texts)

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


def rb_baseline(df, label_col, text_col, progress):
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    texts = df[text_col].to_list()
    true_labels = df[label_col].to_list()

    score_threshold = 0.94
    max_features = 1000
    min_freq = 3
    use_stemming = True
    ngram_range = (1, 2)

    classifier = RBClassifier(
        min_freq=min_freq,
        max_features=max_features,
        score_threshold=score_threshold,
        use_stemming=use_stemming,
        ngram_range=ngram_range,
    )
    classifier.fit(texts, true_labels)

    predictions = classifier.predict(texts)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    num_rules = len(classifier.get_learned_rules())

    top_rules = classifier.get_learned_rules()[:10]

    rules_text = "TOP 10 LEARNED RULES:\n"
    rules_text += f"{'Rank':<6} {'Pattern':<30} {'Score':<10}\n"
    rules_text += f"{'-' * 50}\n"

    for i, (pattern, score) in enumerate(top_rules, 1):
        rules_text += f"{i:<6} {pattern:<30} {score:<10.4f}\n"

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_rules": num_rules,
        "top_rules": str(top_rules) if top_rules is not None else "",
        "rules_text": str(rules_text) if rules_text is not None else "",
        "experiment": "rule_based",
        "label_col": label_col,
        "text_col": text_col,
    }
