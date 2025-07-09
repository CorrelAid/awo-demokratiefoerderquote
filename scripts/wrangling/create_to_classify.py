import polars as pl
import click
import markdown
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from lib import stopwords

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

    df = df.rename({"description": "description_raw"})

    df = df.with_columns(
        pl.col("description_raw")
        .map_elements(md, return_dtype=pl.String)
        .alias("description_md")
    )

    df = df.with_columns(
        pl.col("description_md")
        .map_elements(markdown.markdown, return_dtype=pl.String)
        .alias("description_html")
    )

    df = df.with_columns(
        pl.format(
            """
        {}

        ---

        Förderbereich: {}

        Förderberechtigte: {}""",
            pl.col("description_md"),
            pl.col("funding_area").fill_null([]).list.join(", ", ignore_nulls=False),
            pl.col("eligible_applicants")
            .fill_null([])
            .list.join(", ", ignore_nulls=False),
        ).alias("description_cats_md")
    )

    def remove_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        return " ".join(soup.get_text().split())

    df = df.with_columns(
        pl.col("description_html")
        .map_elements(remove_tags, return_dtype=pl.String)
        .alias("description_text_only")
    )

    def remove_title_from_description(row):
        title = row["title"]
        description = row["description_text_only"]
        return description.replace(title, "")

    def remove_title_stopwords_from_description(row, stopwords=stopwords):
        desc = remove_title_from_description(row)
        temp = desc
        for stopword in stopwords:
            temp = temp.strip().replace(stopword.strip(), "")
        return temp

    df = df.with_columns(
        pl.format(
            """{} (Förderbereich: {}; Förderberechtigte: {}): {}""",
            pl.col("title"),
            pl.col("funding_area").fill_null([]).list.join(", ", ignore_nulls=False),
            pl.col("eligible_applicants")
            .fill_null([])
            .list.join(", ", ignore_nulls=False),
            pl.struct(["title", "description_text_only"]).map_elements(
                remove_title_from_description, return_dtype=pl.String
            ),
        ).alias("description_title_cats_compact")
    )

    df = df.with_columns(
        pl.format(
            """
{} (Förderbereich: {}; Förderberechtigte: {}): {}
        """,
            pl.col("title"),
            pl.col("funding_area").fill_null([]).list.join(", ", ignore_nulls=False),
            pl.col("eligible_applicants")
            .fill_null([])
            .list.join(", ", ignore_nulls=False),
            pl.struct(["title", "description_text_only"]).map_elements(
                remove_title_stopwords_from_description, return_dtype=pl.String
            ),
        ).alias("description_title_cats_compact_wo_section_stopwords")
    )

    # Final check for None/empty values in processed text columns
    text_columns = [
        "description_raw",
        "description_md",
        "description_html",
        "description_cats_md",
        "description_text_only",
        "description_title_cats_compact",
        "description_title_cats_compact_wo_section_stopwords",
    ]

    for col in text_columns:
        if col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            empty_count = df.select((pl.col(col) == "").sum()).item()
            if null_count > 0 or empty_count > 0:
                print(
                    f"Final check - Column '{col}' has {null_count} null values and {empty_count} empty values"
                )

    df = df.with_columns(
        [
            pl.col("eligible_applicants").list.join(", "),
            pl.col("funding_area").list.join(", "),
            pl.col("funding_type").list.join(", "),
            pl.col("funding_location").list.join(", "),
        ]
    )

    df.write_csv(f"{output_folder}/to_classify.csv")


if __name__ == "__main__":
    process()
