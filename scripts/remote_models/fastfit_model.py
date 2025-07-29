import modal
from dotenv import load_dotenv
import os
import subprocess
import polars as pl
from datasets import Dataset, ClassLabel, DatasetDict
from lib.experiment_config import data_path, test_size, n_splits, random_state
from fastfit import FastFitTrainer, sample_dataset
import numpy as np
from fastfit import FastFit
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer, pipeline
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import logging
import optuna
import json

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
        "transformers",
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


@app.function(
    gpu=f"A100-80GB:{N_GPU}",
    timeout=50000,
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
    def objective(trial):
        num_train_epochs = trial.suggest_int("num_train_epochs", 30, 50)
        batch_size = trial.suggest_int("batch_size", 16, 32, step=16)
        clf_loss_factor = trial.suggest_float(
            "clf_loss_factor", 0.05, 0.15
        )  # The factor to scale the classification loss.
        num_repeats = trial.suggest_int(
            "num_repeats", 3, 5
        )  # The number of times to repeat the queries and docs in every batch.

        if smoke_run:
            num_splits = 2
            trial.set_user_attr(
                "model_name", "sentence-transformers/paraphrase-mpnet-base-v2"
            )
            trial.set_user_attr("max_length", 128)
        else:
            num_splits = n_splits
            trial.set_user_attr("model_name", "mixedbread-ai/mxbai-embed-large-v1")
            trial.set_user_attr("max_length", 512)

        texts_array = np.array(texts)
        labels_array = np.array(true_labels)

        fold_metrics = []
        sss = StratifiedShuffleSplit(
            n_splits=num_splits, test_size=test_size, random_state=random_state
        )
        print("Starting training")
        print("Number of splits: ", num_splits)
        for fold, (train_idx, test_idx) in enumerate(
            sss.split(texts_array, labels_array)
        ):
            print(f"\n----Starting fold {fold} of {num_splits}-----\n")
            X_train, X_test = texts_array[train_idx], texts_array[test_idx]

            y_train, y_test = labels_array[train_idx], labels_array[test_idx]

            train = Dataset.from_dict({"text": X_train, "label": y_train})
            test = Dataset.from_dict({"text": X_test, "label": y_test})
            dataset = DatasetDict({"train": train, "test": test})

            if smoke_run:
                dataset["train"] = sample_dataset(
                    dataset["train"], num_samples_per_label=2, label_column="label"
                )

            dataset = convert_labels_to_strings(dataset)

            trainer = FastFitTrainer(
                model_name_or_path=trial.user_attrs["model_name"],
                label_column_name="label",
                text_column_name="text",
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                max_text_length=trial.user_attrs["max_length"],
                dataloader_drop_last=False,
                num_repeats=num_repeats,
                clf_loss_factor=clf_loss_factor,
                fp16=True,
                dataset=dataset,
            )

            model = trainer.train()

            model = FastFit(model.config, encoder=model.encoder)
            tokenizer = AutoTokenizer.from_pretrained(trial.user_attrs["model_name"])

            predictions, _ = classifier(dataset["test"]["text"], model, tokenizer)

            metrics = compute_metrics(predictions, dataset["test"]["label"])

            print(metrics)

            fold_metrics.append(metrics)

        fold_metrics = pl.DataFrame(fold_metrics)
        stats = fold_metrics.describe()

        metrics = ["accuracy", "precision", "recall", "f1"]
        stats_dict = {metric: {} for metric in metrics}

        print(stats)

        for metric in metrics:
            stats_dict[metric]["mean"] = stats.filter(pl.col("statistic") == "mean")[
                metric
            ].to_numpy()[0]
            stats_dict[metric]["std"] = stats.filter(pl.col("statistic") == "std")[
                metric
            ].to_numpy()[0]

            trial.set_user_attr(f"mean_{metric}", stats_dict[metric]["mean"])
            trial.set_user_attr(f"std_{metric}", stats_dict[metric]["std"])

        trial.report(stats_dict["f1"]["mean"], fold)

        return stats_dict["f1"]["mean"]

    if smoke_run:
        n_trials = 2
    else:
        n_trials = 10

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial

    best_params = study.best_params

    train = Dataset.from_dict({"text": texts, "label": true_labels})

    test = Dataset.from_dict({"text": texts[:1], "label": true_labels[:1]})

    dataset = DatasetDict({"train": train, "test": test})

    dataset = convert_labels_to_strings(dataset)

    # trainer = FastFitTrainer(
    #     model_name_or_path=best_trial.user_attrs["model_name"],
    #     label_column_name="label",
    #     text_column_name="text",
    #     num_train_epochs=best_params["num_train_epochs"],
    #     per_device_train_batch_size=best_params["batch_size"],
    #     per_device_eval_batch_size=best_params["batch_size"],
    #     max_text_length=best_trial.user_attrs["max_length"],
    #     dataloader_drop_last=False,
    #     num_repeats=best_params["num_repeats"],
    #     optim="adafactor",
    #     clf_loss_factor=best_params["clf_loss_factor"],
    #     fp16=True,
    #     dataset=dataset,
    # )

    # final_classifier = trainer.train()

    results = (
        best_params,
        {
            "best_params": best_params,
            "f1": best_trial.user_attrs.get("mean_f1", 0),
            "accuracy": best_trial.user_attrs.get("mean_accuracy", 0),
            "precision": best_trial.user_attrs.get("mean_precision", 0),
            "recall": best_trial.user_attrs.get("mean_recall", 0),
            "f1_std": best_trial.user_attrs.get("std_f1", 0),
            "accuracy_std": best_trial.user_attrs.get("std_accuracy", 0),
            "precision_std": best_trial.user_attrs.get("std_precision", 0),
            "recall_std": best_trial.user_attrs.get("std_recall", 0),
            "trial_number": study.best_trial.number,
            "total_trials": len(study.trials),
        },
    )

    return results


@app.local_entrypoint()
def main():
    df = pl.read_parquet(data_path)
    best_params, stats = train_model.remote(
        df["description_title_cats_compact"].to_list(),
        df["label_original"].to_list(),
        smoke_run=False,
    )
    with open("model_results/fastfit_stats.json", "w") as f:
        json.dump(stats, f)
    with open("models/fastfit_best_params.json", "w") as f:
        json.dump(best_params, f)
