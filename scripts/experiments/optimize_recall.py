import polars as pl
from lib.tfidf_mlp_optuna import tfidf_mlp_optuna
from lib.experiment_config import n_splits, random_state, test_size
from lib.viz import display_top_results
from rich.progress import Progress
from rich.console import Console
import json
from datetime import datetime

experiment_name = "tfidf_mlp_recall"

label_col = "label_original"
text_col = "description_title_cats_compact"

date = datetime.now().strftime("%H_%d_%m_%Y")

with Progress() as progress:
    df = pl.read_parquet("data/labeled/25_07/to_classify.parquet")
    res = tfidf_mlp_optuna(
        df,
        label_col,
        text_col,
        progress=progress,
        mode="recall_precision",
        n_trials=100,
    )


res["date"] = date
best_model = res["model"]
model_path = f"models/{experiment_name}.pkl"
best_model.save(model_path)
best_params = res["best_params"]
with open(f"models/{experiment_name}_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
res.pop("model", None)
res.pop("best_params", None)

pl.DataFrame(res).write_csv(f"model_results/opt_recall/{experiment_name}.csv")
