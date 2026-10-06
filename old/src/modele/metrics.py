import polars as pl
from src import logger
from unidecode import unidecode


# Define a function to remove accents
def remove_accents(text: str) -> str:
    return unidecode(text)


def _clean_string(df, cols=None):
    if cols is None:
        cols = [col for col, dtype in df.schema.items() if dtype == pl.Utf8]
    
    logger.info(f"Cleaning string format of columns {cols}")

    for col in cols:
        df = df.with_columns(pl.col(col).str.normalize(form="NFKD").str.replace(r"[^\w\s]", ""))
        df = df.with_columns(pl.col(col).str.to_uppercase())
        df = df.with_columns(pl.col(col).map_elements(remove_accents, return_dtype=pl.Utf8))
    
    return df


def _metrics(df, col_citrus, col_bodacc, key="num_bodacc"):

    if key not in df.columns:
        logger.warning("Calculate metrics without key. Aborting.")
        logger.warning(f"Key {key} absent. Available column names are {df.columns}")
        return None
    
    df = df.select([key, col_citrus, col_bodacc])
    
    col_assert = col_citrus + "_assert"

    # Converting data types
    col_types = df.schema
    if col_types[col_citrus] != col_types[col_bodacc]:
        logger.warning(f"Columns {col_citrus} and {col_bodacc} are of different types : {col_types[col_citrus]} and {col_types[col_bodacc]}")

        # If both are numeric, convert to float32.
        if col_types[col_bodacc].is_numeric() and col_types[col_citrus].is_numeric():
            logger.warning(f"Columns {col_citrus} and {col_bodacc} are both numeric - casting them to Float")
            df = df.cast({col_bodacc: pl.Float32, col_citrus: pl.Float32})

        elif col_types[col_bodacc].is_temporal() and col_types[col_citrus].is_temporal():
            logger.warning(f"Columns {col_citrus} and {col_bodacc} are both temporal - casting them to Date")
            df = df.cast({col_bodacc: pl.Date, col_citrus: pl.Date})

        # Else, convert all to string
        else:
            logger.warning(f"Columns {col_citrus} and {col_bodacc} are not numeric - casting them to String")
            df = df.cast({col_bodacc: pl.Utf8, col_citrus: pl.Utf8})

    col_types = df.schema
    if col_types[col_citrus].is_numeric():
        logger.info(f"Columns {col_citrus} and {col_bodacc} are numeric - equal with 0.1 precision")
        metrics_df = (
            df.with_columns(
                (pl.col(col_citrus) - pl.col(col_bodacc).abs() < 0.1).alias(col_assert)
            )
            .select([key, col_assert])
        )
    elif col_types[col_citrus].is_temporal():
        logger.info(f"Columns {col_citrus} and {col_bodacc} are temporal - equal with 0 precision")
        metrics_df = (
            df.with_columns(
                (pl.col(col_citrus)==pl.col(col_bodacc)).alias(col_assert)
            )
            .select([key, col_assert])
        )
    else:
        logger.info(f"Columns {col_citrus} and {col_bodacc} are string - asserting full equality")
        df = _clean_string(df, cols=[key, col_citrus, col_bodacc])
        metrics_df = (
            df.with_columns(
                (pl.col(col_citrus) == pl.col(col_bodacc)).alias(col_assert)
            )
            .select([key, col_assert])
        )

    return metrics_df


def _add_metrics(previous_metrics_df, new_metrics_df, key="num_bodacc"):
    logger.info(f"Merging with previous metrics based on column {key}")

    full_metrics_df = previous_metrics_df.join(new_metrics_df, on=key, how="right")
    return full_metrics_df


def calculate_metrics(df):
    key = "num_bodacc"
    metrics_df = df.select(key)

    for col_to_assert in [col[:-10] for col in df.columns if col[-10:] == "_apibodacc"]:
        col_citrus = col_to_assert
        col_bodacc = col_to_assert + "_apibodacc"
        metrics_df = _add_metrics(
            metrics_df,
            _metrics(df, key=key, col_citrus=col_citrus, col_bodacc=col_bodacc),
            key=key
        )

    return metrics_df


def filter_metrics_bad(df, res, col_citrus):
    key = "num_bodacc"
    col_assert = col_citrus + "_assert"
    col_bodacc = col_citrus + "_apibodacc"
    return df.filter(pl.col(key).is_in(res.filter(~pl.col(col_assert))[key].to_list())).select(["lien_bodacc", key, col_citrus, col_bodacc])


def print_df_md(df):
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        tbl_rows=-1,   # show all rows; default caps at 10
    ):
        print(df)


def filter_metrics_bad_md(df, res, col_citrus):
    return print_df_md(filter_metrics_bad(df, res, col_citrus))
