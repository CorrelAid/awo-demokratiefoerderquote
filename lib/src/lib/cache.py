import redis
import pickle
import hashlib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset
from lib.experiment_config import (
    n_splits,
    random_state,
    test_size,
    augment_model,
    augment_base_url,
)
from lib.data_augmentation import extend_pipeline
from dotenv import load_dotenv
import os

load_dotenv()


AUGMENT_API_KEY = os.getenv("OR_KEY")


def _generate_cache_key(text_col, label_col, augment_factor):
    key_data = (
        str(text_col).encode("utf-8"),  # convert to bytes
        str(label_col).encode("utf-8"),  # convert to bytes
        str(augment_factor).encode("utf-8"),  # convert to bytes
    )
    raw_key = b"|".join(key_data)
    return hashlib.sha256(raw_key).hexdigest()


def make_folds_dataframe(
    texts_array,
    labels_array,
    text_col,
    label_col,
    augment_factor,
    return_np=True,
    logging=False,
):
    r = redis.Redis(host="localhost", port=6379, db=0)

    cache_key = f"folds:{_generate_cache_key(text_col, label_col, augment_factor)}"

    cached_data = r.get(cache_key)
    if cached_data:
        if logging:
            print(
                f"Loading folds from cache for {text_col}, {label_col}, {augment_factor}"
            )
        return pickle.loads(cached_data)

    print(
        f"Computing folds. No entry for {text_col}, {label_col}, {augment_factor} in cache."
    )
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    folds = []

    for fold, (train_idx, test_idx) in enumerate(sss.split(texts_array, labels_array)):
        X_train, X_test = texts_array[train_idx], texts_array[test_idx]
        y_train, y_test = labels_array[train_idx], labels_array[test_idx]

        if augment_factor > 1:
            texts = X_train.tolist()
            labels = y_train.tolist()

            dataset = Dataset.from_dict({"text": texts, "label": labels})
            extended_dataset = extend_pipeline(
                dataset,
                augment_factor - 1,
                augment_model,
                augment_base_url,
                AUGMENT_API_KEY,
                labeled=True,
            )
            if return_np:
                X_train, y_train = (
                    np.array(extended_dataset["text"]),
                    np.array(extended_dataset["label"]),
                )
            else:
                X_train, y_train = extended_dataset["text"], extended_dataset["label"]

        folds.append(
            {
                "fold": fold,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
        )

    # Cache the computed folds
    r.set(cache_key, pickle.dumps(folds))
    print("Folds computed and cached.")

    return folds
