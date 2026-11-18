from pathlib import Path

import pandas as pd


DATA_DIR = Path.home() / "Desktop" / "datos_futbol_limpios"


def cargar_datos() -> dict[str, pd.DataFrame]:
    archivos_csv = sorted(DATA_DIR.glob("*.csv"))
    if not archivos_csv:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {DATA_DIR}")

    return {
        archivo.stem: pd.read_csv(archivo)
        for archivo in archivos_csv
    }


def preparar_goles_mundial(
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    claves_partido = ["date", "home_team", "away_team"]
    resultados = dataframes["resultados_limpio"]
    partidos_mundial = resultados.loc[
        resultados["tournament"].eq("FIFA World Cup"), claves_partido
    ].drop_duplicates()
    df_mundial = dataframes["marcadores_limpio"].merge(
        partidos_mundial, on=claves_partido, how="inner"
    )

    minuto_base = (
        df_mundial["minute"].astype("string").str.split("+").str[0].str.strip()
    )
    df_mundial["minuto_num"] = pd.to_numeric(
        minuto_base, errors="coerce"
    ).astype("Int64")
    no_interpretados = minuto_base.notna() & df_mundial["minuto_num"].isna()
    print(f"Minutos no interpretados en Mundiales: {no_interpretados.sum()}")
    return df_mundial
