import polars as pl
import click
from lib.wrangling_helpers import create_classify_cols
import polars as pl
import requests
import zipfile
import io
from lib import data_url
from pathlib import Path

"""
Newly created columns:
- description_raw: Original description column containing html as it was scraped from the website section
- description_md: HTML content converted to Markdown format
- description_html: Markdown content converted back to HTML format so that it only contains relevant HTML tags 
- description_cats_md: Markdown description with funding area and eligible applicants appended 
- description_text_only: HTML tags removed, plain text only 
- description_title_cats_compact: Formatted string with title, funding categories, and description (title removed from description)
- description_title_cats_compact_wo_section_stopwords: Same as above but with text section titles ("Kurztext", "Volltext", "Ziel und Gegenstand") removed
"""

# uv run python scripts/wrangling/finalize.py data/labeled/25_07/extended.csv data/labeled/25_07

manuall_retrieved_extend = [
    "7c8f48d1057922942222d6fe9088a26b",
    "50214062cbf23020bcf1b2e5bbc729ee",
    "8a71de46a1b34526fbd4c2acb5a00846",
    "cb926c28e34be88ff7a44d3407398c52",
]


@click.command()
@click.argument(
    "folder",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    metavar="FOLDER",
)
def process(folder):
    lst = []
    for method in ["majority", "equal", "original"]:
        input_path = Path(folder) / f"consolidated_labels_{method}.csv"
        df_labeled = (
            pl.read_csv(input_path)
            .rename({"label": f"label_{method}"})
            .extend(
                pl.DataFrame(
                    {
                        "id_hash": manuall_retrieved_extend,
                        f"label_{method}": [1] * len(manuall_retrieved_extend),
                    }
                )
            )
            .unique("id_hash", keep="last")
        )
        lst.append(df_labeled)

    legacy_path = "data/labeled/19_06/legacy.json"
    df_legacy = pl.read_json(legacy_path)

    print("preparin legacy")
    df_legacy = create_classify_cols(df_legacy, "description").drop("label")

    response = requests.get(data_url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_name = "min_data_format.csv"
        with z.open(file_name) as f:
            df_remote = pl.read_csv(f)
    print("preparin remote")

    df_remote = create_classify_cols(df_remote, "description")

    df = df_legacy.extend(df_remote.select(df_legacy.columns)).unique(
        "id_hash", keep="last"
    )

    df = df.filter(
        pl.col("id_hash").is_in(lst[0]["id_hash"].to_list() + manuall_retrieved_extend)
    ).sort("id_hash")

    for df_labeled, consolidation_method in zip(lst, ["majority", "equal", "original"]):
        df = df.join(df_labeled, on="id_hash", how="left")

        assert len(set(df["id_hash"].to_list())) == len(df["id_hash"].to_list())

    print(f"final shape: {df.shape}")
    print(f"final cols: {df.columns}")

    df.write_parquet(f"{folder}/to_classify.parquet")


if __name__ == "__main__":
    process()
