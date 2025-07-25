import polars as pl
import click
from lib.wrangling_helpers import create_classify_cols
from lib import data_url
import json
import requests
import zipfile
import io
import numpy as np


@click.command()
@click.argument(
    "input_add_manual",
    type=click.Path(exists=True),
    required=True,
    metavar="INPUT_FILE_ADD_MANUAL_JSON",
)
@click.argument(
    "input_to_extend",
    type=click.Path(exists=True),
    required=True,
    metavar="INPUT_FILE_TO_EXTEND_CSV",
)
@click.argument(
    "output_folder",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    metavar="OUTPUT_FOLDER",
)
def process(input_add_manual, input_to_extend, output_folder):
    with open(input_add_manual, "r") as f:
        add_lst = json.load(f)

    response = requests.get(data_url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_name = "min_data_format.csv"
        with z.open(file_name) as f:
            df_all = pl.read_csv(f)

    df_filtered = df_all.filter(pl.col("id_hash").is_in(add_lst))

    df_filtered = create_classify_cols(df_filtered, "description")
    df_filtered = df_filtered.with_columns(
        label=np.array([1 for i in range(len(df_filtered))])
    )["label", "id_hash"]

    df_to_extend = pl.read_csv(input_to_extend)["label", "id_hash"]

    df = df_to_extend.extend(df_filtered)
    print(df)


if __name__ == "__main__":
    process()
