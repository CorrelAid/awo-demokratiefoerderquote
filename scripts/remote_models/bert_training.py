import modal
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
import polars as pl
import torch
import gc
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    AutoConfig,
    TrainingArguments,
)
import torch.nn.functional as F
from lib.experiment_config import cpt_bert, n_splits, random_state, test_size, data_path
from lib.cache import make_folds_dataframe
import json


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


class CustomTrainer(Trainer):
    def __init__(self, class_weights, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            logits = outputs["logits"]
            ce_loss = F.cross_entropy(
                logits, labels, weight=self.class_weights, reduction="none"
            )
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            loss = focal_loss.mean()
        return (loss, outputs) if return_outputs else loss


def calculate_weights_from_ratio(labels, ratio):
    counts = np.bincount(labels)
    if counts[0] > counts[1]:
        w = torch.tensor([1.0, ratio], dtype=torch.float32)
    else:
        w = torch.tensor([ratio, 1.0], dtype=torch.float32)
    return w.to("cuda" if torch.cuda.is_available() else "cpu")


def apply_threshold_predictions(logits, threshold):
    probs = F.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return (probs >= threshold).astype(int)


def print_metrics_summary(all_metrics, metric_name="Metrics"):
    print(f"\n📊 {metric_name} Summary:")
    print("=" * 50)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [m[metric] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.capitalize():>10}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"{'':>10}  Per fold: {[f'{v:.4f}' for v in values]}")


load_dotenv()


image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("lib")
)
app = modal.App(name="awo_demokratiefoerderrechner_bert", image=image)


@app.function(gpu="L40S:1", timeout=50000, secrets=[modal.Secret.from_dotenv(__file__)])
def train_model(folds, label_col, text_col):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    lr = 2.5e-5
    bs = 8
    wd = 0.01
    wm = 0.2
    ep = 2
    wr = 5.0
    gm = 2.0
    threshold = 0.6
    hidden_dropout_prob = 0.3
    attention_probs_dropout_prob = 0.2

    login(token=os.environ["HF_TOKEN"], new_session=True, add_to_git_credential=False)
    model_name = f"correlaid/{cpt_bert}"
    print(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=512
        )

    for i, fold in enumerate(folds):
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]

        train_data = [{"text": x, "labels": y} for x, y in zip(X_train, y_train)]
        val_data = [{"text": x, "labels": y} for x, y in zip(X_test, y_test)]

        tok_tr = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
        tok_val = Dataset.from_list(val_data).map(tokenize_fn, batched=True)

        folds[i] = {
            "train": tok_tr,
            "val": tok_val,
            "val_lbl": y_test,
            "fold_idx": i,
        }

    fold_metrics, f1s = [], []
    for f in folds:
        cw = calculate_weights_from_ratio(f["train"]["labels"], wr)
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.num_labels = 2

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, trust_remote_code=True
        )

        args = TrainingArguments(
            output_dir="./results",
            learning_rate=lr,
            per_device_train_batch_size=bs,
            num_train_epochs=ep,
            weight_decay=wd,
            warmup_ratio=wm,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            logging_steps=50,
            fp16=True,
        )
        trainer = CustomTrainer(
            class_weights=cw,
            gamma=gm,
            model=model,
            args=args,
            train_dataset=f["train"],
        )
        trainer.train()
        logits = trainer.predict(f["val"]).predictions
        preds = apply_threshold_predictions(logits, threshold)

        metrics = {
            "accuracy": accuracy_score(f["val_lbl"], preds),
            "precision": precision_score(f["val_lbl"], preds, zero_division=0),
            "recall": recall_score(f["val_lbl"], preds, zero_division=0),
            "f1": f1_score(f["val_lbl"], preds, zero_division=0),
        }
        print(metrics)
        fold_metrics.append(metrics)
        f1s.append(metrics)

        del trainer, model
        clear_gpu_memory()

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
def main(smoke_run=False):
    text_col = "description_title_cats_compact"
    label_col = "label_original"
    df = pl.read_parquet(data_path)

    texts_array = df[text_col].to_numpy()
    labels_array = df[label_col].to_numpy()

    folds = make_folds_dataframe(
        texts_array,
        labels_array,
        label_col="label_original",
        text_col="description_title_cats_compact",
        augment_factor=1,
        return_np=False,
    )

    results = train_model.remote(folds, text_col=text_col, label_col=label_col)
    with open("model_results/bert_stats.json", "w") as f:
        json.dump(results, f)
