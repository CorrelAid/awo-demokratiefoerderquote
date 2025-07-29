import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns
from rich.table import Table
from rich.console import Console


def count_plot(df: pl.DataFrame, col: str, missing_label: str = "Missing"):
    raw = df[col].to_list()

    lst = [x if x is not None else missing_label for xs in raw for x in (xs or [None])]

    # 3) build a Polars table of counts
    counts_df = (
        pl.DataFrame({col: lst})
        .with_columns(pl.col(col).fill_null(missing_label))
        .group_by(col)
        .count()
        .rename({"count": "n"})
        .sort("n", descending=True)
    )

    # 4) plot
    fig, ax = plt.subplots()
    sns.barplot(
        data=counts_df.to_pandas(), y=col, x="n", order=counts_df[col].to_list(), ax=ax
    )
    ax.set_title(f"Counts of {col} (incl. '{missing_label}')")
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    plt.tight_layout()
    plt.show()


def display_top_results(df, model, console, k=5, sort_by="f1"):
    table = Table(title=f"{model} Top 5 Results by F1 Score", show_lines=True)

    columns_to_show = ["accuracy", "precision", "recall", "f1", "label_col", "text_col"]

    for col in columns_to_show:
        table.add_column(
            col, justify="right" if col not in ["label_col", "text_col"] else "left"
        )

    for row in df.sort(sort_by, descending=True).head(k).iter_rows(named=True):
        table.add_row(
            *(
                f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])
                for col in columns_to_show
            )
        )

    console.print(table)
