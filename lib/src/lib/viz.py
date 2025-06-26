import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns


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
