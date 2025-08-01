import dspy
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dspy.primitives import Example
from lib.experiment_config import (
    code_book_path,
    gen_llm_url,
    random_state,
    augment_model,
)
import sys
from contextlib import contextmanager
import tqdm

tqdm.tqdm.disable = True

load_dotenv()

print(f"API Key exists: {os.getenv('OR_KEY') is not None}")


# dspy uses plain print statements at some points...
@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def validate_answer(example, pred, trace=None):
    return example.label == pred.label


with open(code_book_path, "r") as file:
    codebook_text = file.read()

ngram_range = (1, 2)
min_df = 1
max_features = 1000


inst = f"""You are a binary classifier that has received the following codebook to classify german funding programs based on whether they are funding democracy. 
Once one or more of the following categories are present in the funding program, it is classified as democracy funding:

{codebook_text}

You will receive a funding program text and you have to classify it as either 0 (No Democracy Funding) or 1 (Democracy Funding).
"""


models = [augment_model]


def icl(df, label_col, text_col, progress, t):
    task_2 = progress.add_task(
        f"Running ICL for {label_col} and {text_col} and {t}...",
        total=len(models),
    )
    dspy.disable_logging()
    dspy.disable_litellm_logging()

    metrics = []

    for model_name in models:
        lm = dspy.LM(
            f"openai/{model_name}",
            api_key=os.getenv("OR_KEY"),
            base_url=gen_llm_url,
            temperature=0.0,
            cache=False,
        )

        dspy.configure(lm=lm)
        program = dspy.Predict(
            dspy.Signature("funding_program_text: str -> label: int", inst)
        )
        test_size = 0.7

        df = df.sample(fraction=1.0, seed=random_state)

        test, train = (
            df.head(int(len(df) * test_size)),
            df.tail(int(len(df) * (1 - test_size))),
        )

        X_train, X_test = train[text_col].to_list(), test[text_col].to_list()
        y_train, y_test = train[label_col].to_list(), test[label_col].to_list()

        if t == "icl_mipro":
            size = "medium"
            train_examples = [
                Example(funding_program_text=x[0], label=x[1]).with_inputs(
                    "funding_program_text"
                )
                for x in zip(X_train, y_train)
            ]

            optimizer = dspy.MIPROv2(
                metric=(lambda x, y, trace=None: x.label == y.label),
                num_threads=24,
                auto=size,
            )

            print("compiling")

            compiled = optimizer.compile(
                program, trainset=train_examples, requires_permission_to_run=False
            )
        else:
            compiled = program

        def process_single_item(x):
            result = compiled(funding_program_text=x)
            return int(result.label)

        print("Testing...")
        with ThreadPoolExecutor(max_workers=30) as executor:
            test_data = X_test
            y_pred = list(executor.map(process_single_item, test_data))

        f1 = f1_score(y_test, y_pred, average="binary")
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        accuracy = accuracy_score(y_test, y_pred)

        progress.update(task_2, advance=1)

        metrics.append(
            {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "model_name": model_name,
                "experiment": f"icl_{t}",
                "label_col": label_col,
                "text_col": text_col,
                "model": compiled,
            }
        )
    return metrics


def icl_zero_shot(df, label_col, text_col, progress):
    return icl(df, label_col, text_col, progress, t="zero_shot")


def icl_mipro(df, label_col, text_col, progress):
    return icl(df, label_col, text_col, progress, t="icl_mipro")
