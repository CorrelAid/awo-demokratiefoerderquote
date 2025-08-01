import numpy as np
import fakeredis
import pickle
import pytest
from lib.cache import make_folds_dataframe, _generate_cache_key
from lib.experiment_config import n_splits
from unittest.mock import patch


@pytest.fixture
def sample_data():
    texts = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return texts, labels


@pytest.fixture
def fake_redis_client(monkeypatch):
    fake_redis = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("redis.Redis", lambda *args, **kwargs: fake_redis)
    return fake_redis


@pytest.fixture
def mock_extend_pipeline():
    with patch("lib.data_augmentation.extend_pipeline") as mock:
        mock.side_effect = lambda dataset, *_args, **_kwargs: dataset  # No-op
        yield mock


def test_make_folds_dataframe_computes_and_caches(
    sample_data, fake_redis_client, mock_extend_pipeline
):
    texts_array, labels_array = sample_data

    fake_redis_client.flushall()

    folds = make_folds_dataframe(
        texts_array, labels_array, text_col="text", label_col="label", augment_factor=1
    )

    assert isinstance(folds, list)
    assert len(folds) == n_splits
    assert "X_train" in folds[0]
    assert "y_test" in folds[0]

    key = f"folds:{_generate_cache_key('text', 'label', 1)}"
    cached = fake_redis_client.get(key)
    assert cached is not None

    cached_folds = pickle.loads(cached)
    assert cached_folds[0]["fold"] == folds[0]["fold"]


def test_make_folds_dataframe_uses_cache(
    sample_data, fake_redis_client, mock_extend_pipeline
):
    texts_array, labels_array = sample_data

    key = f"folds:{_generate_cache_key('text', 'label', 1)}"
    fake_result = [
        {"fold": 0, "X_train": [], "y_train": [], "X_test": [], "y_test": []}
    ]
    fake_redis_client.set(key, pickle.dumps(fake_result))

    folds = make_folds_dataframe(
        texts_array, labels_array, text_col="text", label_col="label", augment_factor=1
    )

    assert folds == fake_result
