import polars as pl
import click
from lib.wrangling_helpers import get_annotation, get_hash_id


@click.command()
@click.argument(
    "input_ground_truth",
    type=click.Path(exists=True),
    required=True,
    metavar="INPUT_FILE_GT_JSON",
)
@click.argument(
    "input_to_correct",
    type=click.Path(exists=True),
    required=True,
    metavar="INPUT_FILE_TO_CORRECT_CSV",
)
@click.argument(
    "output_folder",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    metavar="OUTPUT_FOLDER",
)
def process(input_ground_truth, input_to_correct, output_folder):
    gt_df = (
        pl.read_json(input_ground_truth)
        .pipe(get_annotation, "label")
        .pipe(get_hash_id, "id_hash")
        .rename({"label": "gt_label"})
        .select(["id_hash", "gt_label"])
    )

    to_correct_df = pl.read_csv(input_to_correct)

    df = (
        to_correct_df.join(gt_df, on="id_hash", how="left")
        .with_columns(
            # pick gt_label if not null, else use original label
            pl.coalesce(["gt_label", "label"]).alias("label")
        )
        .drop("gt_label")
    )

    df[["id_hash", "label"]].write_csv(f"{output_folder}/corrected_previous_round.csv")


if __name__ == "__main__":
    process()
