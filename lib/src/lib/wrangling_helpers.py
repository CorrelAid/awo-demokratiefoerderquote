import polars as pl
import markdown
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from lib import stopwords


def get_annotation(df, output_col_name):
    df = df.explode("annotations")
    if (
        df["annotations"][0]["result"][0]["value"]["choices"][0] == "yes"
        or df["annotations"][0]["result"][0]["value"]["choices"][0] == "no"
    ):
        pos_str = "yes"
        neg_str = "no"
    else:
        pos_str = "dfy"
        neg_str = "dfn"
    df = df.with_columns(
        pl.col("annotations")
        .struct.field("result")
        .explode()
        .struct.field("value")
        .struct.field("choices")
        .list.first()
        .str.replace(pos_str, "1")
        .str.replace(neg_str, "0")
        .cast(pl.Int64)
        .alias(output_col_name)
    )
    return df


def get_title(df, output_col_name):
    df = df.with_columns(pl.col("data").struct.field("title").alias(output_col_name))
    return df


def get_hash_id(df, output_col_name):
    df = df.with_columns(pl.col("data").struct.field("id_hash").alias(output_col_name))
    return df


def create_classify_cols(df, input_desc_col_name):
    df = df.rename({input_desc_col_name: "description_raw"})

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

    # if column is of type list
    proc_cols = [
        "funding_area",
        "eligible_applicants",
        "funding_type",
        "funding_location",
    ]
    for col in proc_cols:
        dtype = df.schema[col]
        if dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).fill_null(""))
            df = df.rename({col: f"{col}_proc"})
        else:
            df = df.with_columns(
                pl.col(col)
                .fill_null([])
                .list.join(", ", ignore_nulls=False)
                .alias(f"{col}")
            )
            df = df.rename({col: f"{col}_proc"})

    df = df.with_columns(
        pl.format(
            """
        {}

        ---

        Förderbereich: {}

        Förderberechtigte: {}""",
            pl.col("description_md"),
            pl.col("funding_area_proc"),
            pl.col("eligible_applicants_proc"),
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
            pl.col("funding_area_proc"),
            pl.col("eligible_applicants_proc"),
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
            pl.col("funding_area_proc"),
            pl.col("eligible_applicants_proc"),
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
                    f"Final check - Column '{col}' has {null_count} null values and {empty_count} empty values\n"
                )
                # get ids of rows with null or empty values
                print(
                    df.filter(pl.col(col).is_null() | (pl.col(col) == ""))[
                        ["funding_area_proc"]
                    ]
                )
                print("\n")

    for col in proc_cols:
        df = df.rename({f"{col}_proc": col})
        # print sample
        # print(df.select(pl.col(col).sample(5)))
        df = df.with_columns(pl.col(col).str.split(","))
        # print(df.select(pl.col(col).sample(5)))

    return df
