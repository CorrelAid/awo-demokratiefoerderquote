import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from lib.experiment_config import n_splits, random_state, test_size
from lib.MLP import BinaryTfidfMLPClassifier as TfidfMLPClassifier
import optuna
from optuna.samplers import TPESampler
from functools import partial
from lib.data_augmentation import remote_back_translation_augmenter
import asyncio


def objective(trial, df, label_col, text_col, mode):
    """Optuna objective function for hyperparameter optimization"""

    augment_factor = trial.suggest_int("augment_factor", 1, 5)

    texts = df[text_col].to_list()
    true_labels = df[label_col].to_list()

    texts_array = np.array(texts)
    labels_array = np.array(true_labels)

    min_df = trial.suggest_int("min_df", 1, 2)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.05)
    max_features = trial.suggest_categorical("max_features", [5000, 10000])
    use_stemming = True
    ngram_max = 1
    ngram_range = (1, ngram_max)
    hidden_size_1 = trial.suggest_int("hidden_size_1", 96, 208, step=16)
    n_layers = trial.suggest_int("n_layers", 1, 3)

    if n_layers >= 2:
        min_size_2 = max(64, int(hidden_size_1 * 0.6))
        max_size_2 = hidden_size_1

        # Ensure the range is divisible by step=8
        min_size_2 = ((min_size_2 + 7) // 8) * 8  # Round up to nearest multiple of 8
        max_size_2 = (max_size_2 // 8) * 8  # Round down to nearest multiple of 8

        # Ensure min <= max after rounding
        if min_size_2 > max_size_2:
            min_size_2 = max_size_2

        hidden_size_2 = trial.suggest_int(
            "hidden_size_2", min_size_2, max_size_2, step=8
        )
    else:
        hidden_size_2 = None

    if n_layers >= 3:
        min_size_3 = max(64, int(hidden_size_2 * 0.6))
        max_size_3 = hidden_size_2

        # Ensure the range is divisible by step=8
        min_size_3 = ((min_size_3 + 7) // 8) * 8  # Round up to nearest multiple of 8
        max_size_3 = (max_size_3 // 8) * 8  # Round down to nearest multiple of 8

        # Ensure min <= max after rounding
        if min_size_3 > max_size_3:
            min_size_3 = max_size_3

        hidden_size_3 = trial.suggest_int(
            "hidden_size_3", min_size_3, max_size_3, step=8
        )
    else:
        hidden_size_3 = None

    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    epochs = trial.suggest_int("epochs", 40, 100, step=10)
    if mode == "recall":
        threshold = trial.suggest_float("threshold", 0.1, 0.6)
    else:
        threshold = trial.suggest_float("threshold", 0.4, 0.6)

    # Focal loss parameters
    use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
    focal_alpha = (
        trial.suggest_float("focal_alpha", 0.1, 0.5) if use_focal_loss else 0.25
    )
    focal_gamma = (
        trial.suggest_float("focal_gamma", 1.0, 3.0) if use_focal_loss else 2.0
    )

    # Use StratifiedShuffleSplit for HPO (same as final evaluation)
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )

    # Store all metrics for each fold
    fold_metrics = {"f1": [], "accuracy": [], "precision": [], "recall": []}

    try:
        for fold, (train_idx, test_idx) in enumerate(
            sss.split(texts_array, labels_array)
        ):
            X_train, X_test = texts_array[train_idx], texts_array[test_idx]

            y_train, y_test = labels_array[train_idx], labels_array[test_idx]

            if augment_factor > 1:
                loop = asyncio.get_event_loop()
                texts = X_train.tolist()
                labels = y_train.tolist()

                X_train, y_train = loop.run_until_complete(
                    remote_back_translation_augmenter(texts, labels, augment_factor)
                )
                X_train, X_test = np.array(X_train), np.array(y_train)

            classifier = TfidfMLPClassifier(
                min_df=min_df,
                max_features=max_features,
                use_stemming=use_stemming,
                ngram_range=ngram_range,
                hidden_size_1=hidden_size_1,
                hidden_size_2=hidden_size_2,
                hidden_size_3=hidden_size_3,
                learning_rate=learning_rate,
                dropout=dropout,
                epochs=epochs,
                optimizer="adamw",
                threshold=threshold,
                use_focal_loss=use_focal_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                verbose=False,
            )

            classifier.fit(X_train.tolist(), y_train.tolist())
            test_predictions = classifier.predict(X_test.tolist())

            # Calculate all metrics
            fold_f1 = f1_score(y_test, test_predictions, zero_division=0)
            fold_accuracy = accuracy_score(y_test, test_predictions)
            fold_precision = precision_score(y_test, test_predictions, zero_division=0)
            fold_recall = recall_score(y_test, test_predictions, zero_division=0)

            # Store metrics
            fold_metrics["f1"].append(fold_f1)
            fold_metrics["accuracy"].append(fold_accuracy)
            fold_metrics["precision"].append(fold_precision)
            fold_metrics["recall"].append(fold_recall)

            # Report intermediate value for pruning (still use F1 as main metric)
            trial.report(fold_f1, fold)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    except Exception as e:
        print(f"Error during trial: {e}")
        return 0.0

    # Calculate mean metrics
    mean_f1 = np.mean(fold_metrics["f1"])
    mean_accuracy = np.mean(fold_metrics["accuracy"])
    mean_precision = np.mean(fold_metrics["precision"])
    mean_recall = np.mean(fold_metrics["recall"])

    # Store additional metrics as user attributes for later access
    trial.set_user_attr("mean_accuracy", mean_accuracy)
    trial.set_user_attr("mean_precision", mean_precision)
    trial.set_user_attr("mean_recall", mean_recall)
    trial.set_user_attr("std_f1", np.std(fold_metrics["f1"]))
    trial.set_user_attr("std_accuracy", np.std(fold_metrics["accuracy"]))
    trial.set_user_attr("std_precision", np.std(fold_metrics["precision"]))
    trial.set_user_attr("std_recall", np.std(fold_metrics["recall"]))

    recall_weight = 0.8
    precision_weight = 0.2

    if mode == "f1":
        return mean_f1
    elif mode == "recall_precision":
        return recall_weight * mean_recall + precision_weight * mean_precision
    else:
        raise ValueError(f"Invalid mode: {mode}")


def tfidf_mlp_optuna(df, label_col, text_col, progress, mode="f1", n_trials=100):
    """Run Optuna hyperparameter optimization with rich progress bar"""

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=1,
            interval_steps=1,
        ),
        study_name="tfidf_mlp_optimization",
    )

    objective_ = partial(
        objective, df=df, label_col=label_col, text_col=text_col, mode=mode
    )

    task_2 = progress.add_task(
        f"Optimizing hyperparameters for {label_col} and {text_col}...", total=n_trials
    )

    def progress_callback(study, trial):
        progress.update(task_2, advance=1)

    study.optimize(
        objective_,
        n_trials=n_trials,
        callbacks=[progress_callback],
        show_progress_bar=False,
    )

    best_trial = study.best_trial

    texts = df[text_col].to_list()
    true_labels = df[label_col].to_list()

    best_params = study.best_params
    ngram_max = 1
    n_layers = best_params.get("n_layers", 1)

    final_classifier = TfidfMLPClassifier(
        min_df=best_params.get("min_df", 1),
        max_features=best_params.get("max_features", 5000),
        use_stemming=True,
        ngram_range=(1, ngram_max),
        hidden_size_1=best_params.get("hidden_size_1", 160),
        hidden_size_2=best_params.get("hidden_size_2") if n_layers >= 2 else None,
        hidden_size_3=best_params.get("hidden_size_3") if n_layers >= 3 else None,
        learning_rate=best_params.get("learning_rate", 0.035),
        dropout=best_params.get("dropout", 0.42),
        epochs=best_params.get("epochs", 100),
        optimizer="adamw",
        threshold=best_params.get("threshold", 0.52),
        use_focal_loss=best_params.get("use_focal_loss", True),
        focal_alpha=best_params.get("focal_alpha", 0.25),
        focal_gamma=best_params.get("focal_gamma", 2.0),
        verbose=False,
    )

    final_classifier.fit(texts, true_labels)

    results = {
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
        "model": final_classifier,
        "experiment": "tfidf_mlp_optuna",
        "label_col": label_col,
        "text_col": text_col,
        "best_params": best_params,
    }

    return results


##############
# Class weights
# Pass a higher positive‐class weight to your focal loss or to a standard cross‐entropy loss:
# Python

# classifier = TfidfMLPClassifier(
#     …,
#     class_weights={0:1.0, 1: w_pos},
#     …
# )
