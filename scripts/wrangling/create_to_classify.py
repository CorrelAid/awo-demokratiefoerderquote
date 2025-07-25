import polars as pl
import click
from lib.wrangling_helpers import create_classify_cols

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
    df = pl.read_json(input_path)

    df = create_classify_cols(df, "description")

    df.write_csv(f"{output_folder}/to_classify.csv")


if __name__ == "__main__":
    process()
