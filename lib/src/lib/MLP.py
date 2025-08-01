import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted
import torch.nn as nn
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.snowball import GermanStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BinaryMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes: tuple, dropout=0.1, use_logits=False):
        super(BinaryMLP, self).__init__()
        self.use_logits = use_logits
        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_sizes[-1], 1))

        if not use_logits:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def gen_hidden_sizes(hidden_size_1, hidden_size_2, hidden_size_3):
    temp = [hidden_size_1, None]
    if hidden_size_1 != 0:
        temp[1] = hidden_size_2
    if hidden_size_2 != 0:
        temp.append(hidden_size_3)
    return tuple(filter(None, temp))


class BinaryMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_size_1=16,
        hidden_size_2=None,
        hidden_size_3=None,
        learning_rate=0.1,
        dropout=0.1,
        epochs=50,
        gpu=False,
        optimizer="adamw",
        criterion="focal_loss",
        random_state=42,
        init_method="kaiming_uniform",
        verbose=False,
        threshold=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.random_state = random_state
        self.gpu = gpu
        self.init_method = init_method
        self.verbose = verbose
        self.threshold = threshold
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def fit(self, X, y):
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"Binary classifier expects exactly 2 classes, got {len(unique_classes)}"
            )

        hidden_sizes = gen_hidden_sizes(
            self.hidden_size_1, self.hidden_size_2, self.hidden_size_3
        )

        self._le = LabelEncoder().fit(y)
        y_encoded = self._le.transform(y)

        X = check_array(X, accept_sparse=False)
        y_encoded = check_array(y_encoded, ensure_2d=False, accept_sparse=False)

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input data X must be numeric.")

        if y_encoded is None or y_encoded.size == 0:
            raise ValueError("Target variable y cannot be None or an empty array")

        self._input_size = X.shape[1]
        self.classes_ = unique_classes

        use_logits = self.criterion == "focal_loss"
        self._model = BinaryMLP(
            self._input_size, hidden_sizes, self.dropout, use_logits=use_logits
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                if self.init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif self.init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif self.init_method == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                else:
                    raise ValueError(f"Unsupported init_method: {self.init_method}")

        self._model.apply(init_weights)

        # Set up device
        device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
        if self.gpu and torch.cuda.is_available():
            self._model = self._model.to(device)

        # Choose loss function
        if self.criterion == "bce_loss":
            criterion = nn.BCELoss()
        elif self.criterion == "focal_loss":
            criterion = BinaryFocalLoss(
                gamma=self.focal_gamma, alpha=self.focal_alpha, reduction="mean"
            )
        else:
            raise ValueError(
                "Invalid criterion. Supported criterions are 'bce_loss' and 'focal_loss'."
            )

        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=self.learning_rate
            )
        else:
            raise ValueError("Invalid optimizer. Supported optimizers are 'adamw'.")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_encoded, dtype=torch.float32).to(device)

        if self.verbose:
            unique_classes_encoded, counts = np.unique(y_encoded, return_counts=True)
            class_dist = dict(zip(unique_classes_encoded, counts))
            print(f"Class distribution: {class_dist}")
            minority_ratio = min(counts) / max(counts)
            print(f"Minority class ratio: {minority_ratio:.3f}")

        torch.manual_seed(self.random_state)

        for epoch in range(self.epochs):
            self._model.train()
            optimizer.zero_grad()

            outputs = self._model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)

            loss.backward()
            optimizer.step()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.n_features_in_ = X.shape[1]
        self._model.to("cpu")

        return self

    def _predict_proba(self, X):
        check_is_fitted(self)

        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in input ({X.shape[1]}) does not match the number of features in fit ({self.n_features_in_})"
            )

        device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self._model.to(device)
        with torch.no_grad():
            self._model.eval()
            outputs = self._model(X_tensor).squeeze()

            # Convert logits to probabilities if using focal loss
            if self.criterion == "focal_loss":
                probabilities = torch.sigmoid(outputs)
            else:
                probabilities = outputs

        self._model.to("cpu")

        if probabilities.dim() == 0:
            probabilities = probabilities.unsqueeze(0)

        return probabilities.cpu().numpy()

    def predict(self, X):
        probas = self._predict_proba(X)
        predictions = (probas >= self.threshold).astype(int)
        return self._le.inverse_transform(predictions)

    def predict_proba(self, X):
        probas_positive = self._predict_proba(X)
        probas_negative = 1 - probas_positive

        return np.column_stack([probas_negative, probas_positive])


class BinaryTfidfMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_df=2,
        max_features=10000,
        use_stemming=True,
        ngram_range=(1, 2),
        hidden_size_1=16,
        hidden_size_2=None,
        hidden_size_3=None,
        learning_rate=0.1,
        dropout=0.1,
        epochs=50,
        optimizer="adamw",
        threshold=0.5,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        verbose=False,
        smote=False,
        smote_k_neighbors=5,
    ):
        self.min_df = min_df
        self.max_features = max_features
        self.use_stemming = use_stemming
        self.ngram_range = ngram_range
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.optimizer = optimizer
        self.threshold = threshold
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.verbose = verbose
        self.smote = smote
        self.smote_k_neighbors = smote_k_neighbors

        self._initialize_components()

    def _initialize_components(self):
        if self.use_stemming:
            self.stemmer = GermanStemmer()
        else:
            self.stemmer = None

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            token_pattern=r"\b\w+\b",
            lowercase=True,
        )
        if self.smote:
            over = SMOTE(k_neighbors=self.smote_k_neighbors)
            under = RandomUnderSampler()
            steps = [("over", over), ("under", under)]
            self.smote_pipeline = Pipeline(steps=steps)

        criterion = "focal_loss" if self.use_focal_loss else "bce_loss"

        self.classifier = BinaryMLPClassifier(
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            hidden_size_3=self.hidden_size_3,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
            epochs=self.epochs,
            gpu=False,
            optimizer=self.optimizer,
            criterion=criterion,
            random_state=42,
            init_method="kaiming_uniform",
            verbose=self.verbose,
            threshold=self.threshold,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
        )

        self.is_fitted = False

    def get_params(self, deep=True):
        params = {
            "min_df": self.min_df,
            "max_features": self.max_features,
            "use_stemming": self.use_stemming,
            "ngram_range": self.ngram_range,
            "hidden_size_1": self.hidden_size_1,
            "hidden_size_2": self.hidden_size_2,
            "hidden_size_3": self.hidden_size_3,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "threshold": self.threshold,
            "use_focal_loss": self.use_focal_loss,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "verbose": self.verbose,
        }

        if deep and hasattr(self, "classifier"):
            classifier_params = self.classifier.get_params(deep=True)
            for key, value in classifier_params.items():
                params[f"classifier__{key}"] = value

        return params

    def set_params(self, **params):
        classifier_params = {}
        main_params = {}

        for key, value in params.items():
            if key.startswith("classifier__"):
                classifier_params[key[12:]] = value
            else:
                main_params[key] = value

        for key, value in main_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")

        if classifier_params and hasattr(self, "classifier"):
            self.classifier.set_params(**classifier_params)

        self._initialize_components()

        return self

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()

        if self.use_stemming and self.stemmer:
            tokens = word_tokenize(text)
            stemmed_tokens = [
                self.stemmer.stem(token) for token in tokens if token.isalnum()
            ]
            return " ".join(stemmed_tokens)

        return text

    def fit(self, X, y):
        if isinstance(X, (list, np.ndarray)) and len(X) > 0:
            if isinstance(X[0], str):
                texts = X
            else:
                raise ValueError("Expected text input for BinaryTfidfMLPClassifier")
        else:
            raise ValueError("Input X must be a list or array of text strings")

        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(
                f"Binary classifier expects exactly 2 classes, got {len(unique_labels)}: {unique_labels}"
            )

        self.classes_ = unique_labels

        if self.verbose:
            print(f"Training binary classifier with {len(texts)} samples...")
            unique_labels_count, counts = np.unique(y, return_counts=True)
            label_dist = dict(zip(unique_labels_count, counts))
            print(f"Label distribution: {label_dist}")

        processed_texts = [self._preprocess_text(text) for text in texts]

        X_tfidf = self.vectorizer.fit_transform(processed_texts)
        X_dense = X_tfidf.toarray()

        if self.smote:
            X_dense, y = self.smote_pipeline.fit_resample(X_dense, y)

        if self.verbose:
            print(f"TF-IDF feature matrix shape: {X_dense.shape}")

        self.classifier.fit(X_dense, y)

        self.is_fitted = True
        self.n_features_in_ = X_dense.shape[1]

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        if isinstance(X, str):
            X = [X]

        processed_texts = [self._preprocess_text(text) for text in X]
        X_tfidf = self.vectorizer.transform(processed_texts)
        X_dense = X_tfidf.toarray()

        return self.classifier.predict(X_dense)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")

        if isinstance(X, str):
            X = [X]

        processed_texts = [self._preprocess_text(text) for text in X]
        X_tfidf = self.vectorizer.transform(processed_texts)
        X_dense = X_tfidf.toarray()

        return self.classifier.predict_proba(X_dense)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save(self, filepath):
        import pickle
        import os

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted classifier")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Binary classifier saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        import pickle

        with open(filepath, "rb") as f:
            classifier = pickle.load(f)
        print(f"Binary classifier loaded from {filepath}")
        return classifier

    @classmethod
    def from_json(cls, json_path_or_dict, **kwargs):
        if isinstance(json_path_or_dict, str):
            with open(json_path_or_dict, "r") as f:
                data = json.load(f)
        else:
            data = json_path_or_dict

        params = data

        return cls(
            min_df=params.get("min_df", 2),
            max_features=params.get("max_features", 10000),
            hidden_size_1=params.get("hidden_size_1", 16),
            learning_rate=params.get("learning_rate", 0.1),
            dropout=params.get("dropout", 0.1),
            epochs=params.get("epochs", 50),
            threshold=params.get("threshold", 0.5),
            use_focal_loss=params.get("use_focal_loss", True),
            focal_alpha=params.get("focal_alpha", 0.25),
            focal_gamma=params.get("focal_gamma", 2.0),
            **kwargs,
        )
