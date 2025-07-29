import polars as pl
import click
from lib.wrangling_helpers import get_annotation, get_hash_id
from pathlib import Path

# uv run python scripts/wrangling/consolidate_label_versions.py data/labeled/25_07/07_25_awo_coder_1.json  data/labeled/25_07/07_25_awo_coder_2.json data/labeled/25_07/corrected_previous_round.csv data/labeled/25_07


@click.command()
@click.argument(
    "label_round_dfs",
    type=click.Path(exists=True),
    required=True,
    nargs=-1,
    metavar="INPUT_FILE_GT_JSON",
)
@click.argument(
    "output_folder",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    metavar="OUTPUT_FOLDER",
)
def process(label_round_dfs, output_folder):
    input_files = [Path(file) for file in label_round_dfs]

    original = "data/labeled/25_07/corrected_previous_round.csv"
    df_original = pl.read_csv(original)

    dfs = []
    for file in input_files:
        if "csv" in file.suffix:
            df = pl.read_csv(file)["label", "id_hash"]
        else:
            df = pl.read_json(file)
            df = df.pipe(get_annotation, "label").pipe(get_hash_id, "id_hash")[
                "label", "id_hash"
            ]
        dfs.append(df)

    if not dfs:
        return

    all_labels = (
        pl.concat(dfs).group_by("id_hash").agg(pl.col("label").alias("all_labels"))
    )

    def get_majority(series):
        counts = series.value_counts().sort(by="count", descending=True)
        # Get the label with the highest count
        if len(counts) > 0:
            return counts[0, 0]  # Return the label with the highest count
        return None

    df_maj = all_labels.with_columns(
        pl.col("all_labels").map_elements(get_majority).alias("label")
    ).drop("all_labels")

    def get_all_equal(series):
        return len(set(series)) == 1

    df_equal = (
        all_labels.filter(pl.col("all_labels").map_elements(get_all_equal))
        .with_columns(
            pl.col("all_labels")
            .map_elements(lambda x: x[0])
            .alias("label")  # Extract the first label
        )
        .drop("all_labels")
    )  # Drop the 'all_labels' column

    for dict in [
        {"df": df_maj, "name": "majority"},
        {"df": df_equal, "name": "equal"},
        {"df": df_original, "name": "original"},
    ]:
        df = dict["df"]
        assert len(set(df["id_hash"].to_list())) == len(df["id_hash"].to_list())

        output_path = Path(output_folder) / f"consolidated_labels_{dict['name']}.csv"
        df.write_csv(output_path)


if __name__ == "__main__":
    process()
