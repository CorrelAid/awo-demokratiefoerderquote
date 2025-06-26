import polars as pl
import markdown
from markdownify import markdownify as md
import click


@click.command()
@click.argument(
    "input_path", type=click.Path(exists=True), required=True, metavar="INPUT_FILE"
)
@click.argument(
    "output_folder",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    metavar="OUTPUT_FOLDER",
)
def process(input_path, output_folder):
    df_labeled = pl.read_json(input_path)

    df_labeled = df_labeled[["id_hash", "url", "description", "title"]]

    df_labeled = df_labeled.with_columns(
        pl.col("description").map_elements(md, return_dtype=pl.String)
    )

    df_labeled = df_labeled.with_columns(
        pl.col("description").map_elements(markdown.markdown, return_dtype=pl.String)
    )

    df_labeled = df_labeled.rename({"description": "text"})

    df_labeled.write_csv(f"{output_folder}/ls_input.csv")


if __name__ == "__main__":
    process()
