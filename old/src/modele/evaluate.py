from src.bodacc.api import bodacc_api
from src.operation.vente import parse_vente, extract_amount_vente
from src.modele.metrics import calculate_metrics, filter_metrics_bad, _clean_string
from src.utils import parse_to_date_df
from src import logger

import polars as pl


def evaluate_vente(source: str):
    """
    Function to extract information from a list of vente with Bodacc API and compare it back with original file. 
    Original file has to have the following columns : 

    """  
    api = bodacc_api()

    ###########################
    # Import operation Vente
    ###########################
    df = pl.scan_csv(
        source,
        storage_options={
            "aws_endpoint_url": "https://minio.lab.sspcloud.fr",
            "aws_region": "us-east-1",
        },
        credential_provider=pl.CredentialProviderAWS(
            profile_name="service-account",
            region_name="us-east-1",
        ),
    ).collect()

    df = df.with_columns(
        pl.col("lien_bodacc").str.replace("https://www.bodacc.fr/annonce/detail-annonce/", "").str.replace_all("/", "").alias("num_bodacc")
    )

    df = parse_to_date_df(df, "dateEffetComptable")
    logger.info(f"Imported {source} file {df}")

    ###########################
    # Extract info from Bodacc API
    ###########################
    logger.info("Extract info from Bodacc API")

    res_list = []
    for row in df.iter_rows(named=True):
        logger.info(f"{len(res_list)} turn")
        bodac_annonce = api.get_annonce_json(row["num_bodacc"])
        regex_dict = parse_vente(bodac_annonce)
        llm_dict = extract_amount_vente(bodac_annonce)
        full_dict = {"num_bodacc": row["num_bodacc"]} | regex_dict | llm_dict
        res_list.append(full_dict)

    logger.info("Turn it into a Polars dataframe")
    from_bodacc_df = pl.DataFrame(res_list, schema={
        'num_bodacc': pl.Utf8, 
        'anneeCampagne': pl.Int64, 
        'sirenCedant': pl.Utf8, 
        'raisonSocialeCedant': pl.Utf8,
        'sirenBeneficiaire': pl.Utf8,
        'raisonSocialeBeneficiaire': pl.Utf8,
        'dateEffetComptable': pl.Utf8,
        'typeOperation': pl.Utf8,
        'source': pl.Utf8,
        'montantNet': pl.Int64})

    from_bodacc_df = from_bodacc_df.with_columns((pl.col("montantNet")/1000).round(mode="half_away_from_zero")).cast({"montantNet": pl.Int64})

    from_bodacc_df = _clean_string(from_bodacc_df, cols=["raisonSocialeCedant", "raisonSocialeBeneficiaire"])
    from_bodacc_df = parse_to_date_df(from_bodacc_df, "dateEffetComptable", format="%Y-%m-%d")

    from_bodacc_df.write_csv(
        "s3://projet-citrus/data/202607_citrus_bodacc_300ventes_extraction.csv",
        storage_options={
            "aws_endpoint_url": "https://minio.lab.sspcloud.fr",
            "aws_region": "us-east-1",
        },
        credential_provider=pl.CredentialProviderAWS(
            profile_name="service-account",
            region_name="us-east-1",
        )
    )


    ###########################
    # Metrics
    ###########################
    logger.info("Calculate metrics")
    citrus_apibodacc_df = df.join(from_bodacc_df, on="num_bodacc", suffix="_apibodacc")
    res = calculate_metrics(citrus_apibodacc_df)
    logger.info(f"Metrics are \n: {res.mean()}")

    metrics_path = "s3://projet-citrus/data/202607_citrus_bodacc_300ventes_metrics.csv"
    res.write_csv(
        metrics_path,
        storage_options={
            "aws_endpoint_url": "https://minio.lab.sspcloud.fr",
            "aws_region": "us-east-1",
        },
        credential_provider=pl.CredentialProviderAWS(
            profile_name="service-account",
            region_name="us-east-1",
        )
    )

    logger.info(f"Stored metrics as {metrics_path}")
    return from_bodacc_df, res, citrus_apibodacc_df
