import polars as pl

def annuaire(siren_siret: str):
    return f"https://annuaire-entreprises.data.gouv.fr/entreprise/{siren_siret}"


def luhn_checksum(sirene_number):
    sirene_number = str(sirene_number).replace(" ", "")
    # Exception pour La Poste
    if sirene_number == "356000000":
        return 0
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(sirene_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = 0
    checksum += sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10


def is_luhn_valid(sirene_number):
    return luhn_checksum(sirene_number) == 0


def parse_to_date_df(df, col_to_parse="dateEffetComptable", format="%d-%m-%Y"):
    return df.with_columns(
        pl.col(col_to_parse).str.strip_chars().str.to_date(format, strict=False)
    )
