from lib.rule_based import rb_baseline
from lib.tfidf_mlp_optuna import tfidf_mlp_optuna
from lib.ICL import icl_zero_shot, icl_mipro
from lib.viz import display_top_results
from lib.experiment_config import n_splits, random_state, test_size
import polars as pl
from rich.progress import Progress
from rich.console import Console
import json
import os
from collections import Counter
from datetime import datetime

schema = {
    "f1": pl.Float64,
    "precision": pl.Float64,
    "recall": pl.Float64,
    "accuracy": pl.Float64,
    "label_col": pl.Utf8,
    "text_col": pl.Utf8,
    "experiment": pl.Utf8,
    "date": pl.Utf8,
}


label_col = "label_original"

if os.path.exists("model_results/combined_results.csv"):
    with open("model_results/combined_results.csv", "r") as f:
        results_df = pl.read_csv(f)
else:
    results_df = pl.DataFrame(schema=schema)
    results_df.write_csv("model_results/combined_results.csv")


def conduct_experiments(
    experiment_name, experiment_func, console, text_columns, label_col=label_col
):
    total_iterations = len(text_columns)

    results = []
    console.log(
        f"Starting {experiment_name} experiments for {total_iterations} text variants..."
    )
    date = datetime.now().strftime("%H_%d_%m_%Y")
    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Conducting {experiment_name}...", total=total_iterations
        )
        df = pl.read_parquet("data/labeled/25_07/to_classify.parquet").drop_nulls(
            label_col
        )
        for text_col in text_columns:
            res = experiment_func(df, label_col, text_col, progress)
            if isinstance(res, list):
                for x in res:
                    x.update({"date": date})
                results.extend(res)
            else:
                res["date"] = date
                results.append(res)
            progress.update(task, advance=1)

    results = sorted(results, key=lambda x: x["f1"], reverse=True)

    console.log(
        f"Done with {experiment_name} experiments. Displaying top 5 results...\n"
    )

    if experiment_name == "rule_based":
        top_rules = results[0]["top_rules"]
        rules_text = results[0]["rules_text"]
        with open(f"models/{experiment_name}_model.json", "w") as f:
            json.dump({"rules": rules_text, "top_rules": top_rules}, f, indent=2)

        for item in results:
            item.pop("top_rules", None)
            item.pop("rules_text", None)
            item.pop("num_rules", None)
        temp_df = pl.DataFrame(results, schema_overrides=schema)

    elif experiment_name == "optuna_tf_idf_mlp":
        best_model = results[0]["model"]
        model_path = f"models/{experiment_name}.pkl"
        best_model.save(model_path)
        best_params = results[0]["best_params"]
        with open(f"models/{experiment_name}_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        for item in results:
            item.pop("model", None)
            item.pop("best_params", None)
        temp_df = pl.DataFrame(results, schema_overrides=schema)

    elif "icl" in experiment_name:
        results[0]["model"].save(f"models/{experiment_name}.json")

        for item in results:
            item.pop("model", None)
        temp_df = pl.DataFrame(results, schema_overrides=schema)

    display_top_results(temp_df, experiment_name, console)

    results_path = f"model_results/{experiment_name}.csv"

    temp_df.write_csv(results_path)
    console.log(f"Results saved to {results_path}")

    return_df = temp_df[list(schema.keys())]

    return return_df


console = Console()

console.log(f"global config: {n_splits=}, {random_state=}, {test_size=}\n")

for exp in [
    # ("rule_based", rb_baseline),
    ("optuna_tf_idf_mlp", tfidf_mlp_optuna),
    # ("icl_zero_shot", icl_zero_shot),
    # ("icl_mipro", icl_mipro),
]:
    temp = conduct_experiments(
        exp[0],
        exp[1],
        console,
        text_columns=["description_text_only", "description_title_cats_compact"],
    )
    results_df = results_df.extend(temp)


console.log("All experiments completed.")
results_df.write_csv("model_results/combined_results.csv")
