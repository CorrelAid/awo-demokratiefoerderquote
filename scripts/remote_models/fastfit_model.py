import modal
from dotenv import load_dotenv
import os
import subprocess
import polars as pl
from datasets import Dataset, ClassLabel, DatasetDict
from lib.experiment_config import data_path, test_size, n_splits, random_state
from fastfit import FastFitTrainer, sample_dataset
from fastfit.modeling import FastFitConfig
import numpy as np
from fastfit import FastFit
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import optuna
import json
import gc


load_dotenv()

API_KEY = os.getenv("API_KEY")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fast-fit",
        "numpy==1.26.4",
        "python-dotenv",
        "polars",
        "datasets==2.19.0",
        "transformers==4.54.1",
        "optuna",
    )
    .add_local_python_source("lib")
)


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("fastfit_training")

N_GPU = 1
VLLM_PORT = 8000

MINUTES = 60


def compute_metrics(predictions, labels):
    """
    Computes precision, recall, F1 score, and accuracy based on predictions and labels.
    Assumes both predictions and labels are lists of the same length.
    """
    precision = precision_score(labels, predictions, pos_label="dfy", average="binary")
    recall = recall_score(labels, predictions, pos_label="dfy", average="binary")
    f1 = f1_score(labels, predictions, pos_label="dfy", average="binary")
    accuracy = accuracy_score(labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def convert_labels_to_strings(dataset):
    for split in ["train", "test"]:
        dataset[split] = dataset[split].map(
            lambda x: {"label": "dfn" if x["label"] == 0 else "dfy"}
        )
    return dataset


def classifier(texts, model, tokenizer, device="cuda"):
    model = model.to(device)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    inputs.pop("token_type_ids", None)

    outputs = model(**inputs)
    logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).cpu().numpy()

    predicted_class_str = ["dfn" if label == 0 else "dfy" for label in predicted_class]

    return predicted_class_str, probabilities


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


@app.function(
    gpu=f"A100-80GB:{N_GPU}",
    timeout=86400,
    secrets=[modal.Secret.from_dotenv(__file__)],
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
def train_model(
    texts,
    true_labels,
    n_splits=n_splits,
    test_size=test_size,
    random_state=random_state,
    smoke_run=False,
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if smoke_run:
        texts = texts[:50]
        true_labels = true_labels[:50]

    texts_array = np.array(texts)
    labels_array = np.array(true_labels)

    n_epochs = 40
    batch_size = 16
    clf_loss_factor = 0.1
    num_repeats = 4

    model_name = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if smoke_run
        else "mixedbread-ai/mxbai-embed-large-v1"
    )

    max_length = 128 if smoke_run else 512

    sss = StratifiedShuffleSplit(
        n_splits=n_splits if not smoke_run else 2,
        test_size=test_size,
        random_state=random_state,
    )
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(sss.split(texts_array, labels_array)):
        X_train, X_val = texts_array[train_idx], texts_array[test_idx]
        y_train, y_val = labels_array[train_idx], labels_array[test_idx]

        train = Dataset.from_dict({"text": X_train, "label": y_train})
        val = Dataset.from_dict({"text": X_val, "label": y_val})

        dataset = DatasetDict({"train": train, "test": val})
        dataset = convert_labels_to_strings(dataset)

        trainer = FastFitTrainer(
            model_name_or_path=model_name,
            label_column_name="label",
            text_column_name="text",
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_text_length=max_length,
            dataloader_drop_last=False,
            num_repeats=num_repeats,
            clf_loss_factor=clf_loss_factor,
            fp16=True,
            dataset=dataset,
        )

        model = trainer.train()
        model = FastFit(model.config, encoder=model.encoder)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        predictions, _ = classifier(dataset["test"]["text"], model, tokenizer)
        metrics = compute_metrics(predictions, dataset["test"]["label"])
        print(metrics)
        clear_gpu_memory()
        del model
        del trainer
        del tokenizer
        gc.collect()
        fold_metrics.append(metrics)

    df = pl.DataFrame(fold_metrics)
    stats = df.describe()
    metric_keys = ["f1", "accuracy", "precision", "recall"]

    results_dict = {
        key: {
            "mean": stats.filter(pl.col("statistic") == "mean")[key].to_numpy()[0],
            "std": stats.filter(pl.col("statistic") == "std")[key].to_numpy()[0],
        }
        for key in metric_keys
    }

    results = {
        **{f"{k}": v["mean"] for k, v in results_dict.items()},
        **{f"{k}_std": v["std"] for k, v in results_dict.items()},
    }

    print(results)

    return results


@app.local_entrypoint()
def main():
    df = pl.read_parquet(data_path)
    stats = train_model.remote(
        df["description_title_cats_compact"].to_list(),
        df["label_original"].to_list(),
        smoke_run=False,
    )
    with open("model_results/fastfit_stats.json", "w") as f:
        json.dump(stats, f)
