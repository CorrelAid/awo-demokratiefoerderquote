import modal
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset
import polars as pl
import os
import torch
from lib.experiment_config import cpt_bert
import gc
import math
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import (
    AutoModelForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    logging,
)
import numpy as np


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


load_dotenv()

image = modal.Image.debian_slim().pip_install(
    "transformers",
    "datasets",
    "torch",
    "accelerate",
    "sentencepiece",
    "polars",
    "python-dotenv",
    "scikit-learn",
)

app = modal.App(name="dfw_bert_cpt", image=image)


class EarlyStoppingByPerplexityCallback(TrainerCallback):
    def __init__(self, patience=1, tolerance=0.5):
        self.patience = patience
        self.tolerance = tolerance
        self.best_perplexity = float("inf")
        self.stopping_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_loss = metrics.get("eval_loss", float("inf"))
        current_perplexity = compute_perplexity(current_loss)
        if current_perplexity < self.best_perplexity - self.tolerance:
            self.best_perplexity = current_perplexity
            self.stopping_counter = 0
        else:
            self.stopping_counter += 1
            if self.stopping_counter > self.patience:
                control.should_early_stop = True
                control.should_save = True
        return control


def compute_perplexity(loss):
    return np.exp(loss)


MASK_PCT = 0.15
DROPOUT = 0.1

training_args = TrainingArguments(
    output_dir="./results/final_model",
    push_to_hub=True,
    hub_model_id="FundedBert",
    hub_private_repo=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    logging_steps=500,
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    eval_strategy="steps",
    metric_for_best_model="eval_loss",
    optim="adamw_torch",
    warmup_ratio=0.1,
    learning_rate=3e-5,
    lr_scheduler_type="linear",
    fp16=True,
)


@app.function(
    gpu="L40S:1",
    timeout=50000,
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def train_model(texts):
    dataset = Dataset.from_dict({"text": texts})
    all_data = Dataset.from_dict({"text": texts})

    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

    DEVICE = torch.device("cuda")
    model_checkpoint = "deepset/gbert-large"

    login(
        token=os.environ["HF_TOKEN"],
        new_session=True,
        add_to_git_credential=False,
        write_permission=True,
    )
    logging.set_verbosity_error()

    config = BertConfig.from_pretrained(model_checkpoint)
    config.hidden_dropout_prob = DROPOUT
    model = AutoModelForMaskedLM.from_pretrained(
        model_checkpoint, config=config, ignore_mismatched_sizes=True
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, clean_up_tokenization_spaces=True
    )

    def tokenize_function(examples):
        if isinstance(examples["text"], (list, np.ndarray)):
            texts = [str(t) if not isinstance(t, str) else t for t in examples["text"]]
        else:
            texts = str(examples["text"])

        result = tokenizer(texts)
        return result

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    print(tokenized_datasets)

    chunk_size = 512

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("grouping texts")
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=MASK_PCT
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingByPerplexityCallback()],
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print(f">>> Perplexity before: {math.exp(eval_results['eval_loss']):.2f}")

    print("training")
    trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print("Processing full dataset for final training")
    tokenized_all_data = all_data.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    lm_all_data = tokenized_all_data.map(group_texts, batched=True)

    final_training_args = TrainingArguments(
        output_dir="./results/final_model",
        push_to_hub=True,
        hub_model_id=cpt_bert,
        hub_private_repo=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="no",
        optim="adamw_torch",
        warmup_ratio=0.1,
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=lm_all_data,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    trainer.push_to_hub()
    print("✅ done")


@app.local_entrypoint()
def main():
    df = pl.read_parquet("data/labeled/25_07/to_classify.parquet")

    text_col = "description_title_cats_compact"
    texts = df[text_col].to_list()

    train_model.remote(texts)


# {'train_runtime': 13.9947, 'train_samples_per_second': 43.731, 'train_steps_per_second': 2.787, 'train_loss': 0.6291026090964292, 'epoch': 3.0}
# >>> Perplexity before: 2.74
# >>> Perplexity: 1.85
