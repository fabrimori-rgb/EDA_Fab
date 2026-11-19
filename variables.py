from pathlib import Path

import pandas as pd


DATA_DIR = Path.home() / "Desktop" / "datos_futbol_limpios"
ARCHIVOS_CSV = (
    "marcadores_limpio.csv",
    "paises_limpio.csv",
    "penalties_limpio.csv",
    "resultados_limpio.csv",
)


def cargar_datos() -> dict[str, pd.DataFrame]:
    archivos_faltantes = [
        nombre for nombre in ARCHIVOS_CSV if not (DATA_DIR / nombre).is_file()
    ]
    if archivos_faltantes:
        raise FileNotFoundError(
            f"Faltan CSV requeridos en {DATA_DIR}: "
            f"{', '.join(archivos_faltantes)}"
        )

    return {
        Path(nombre).stem: pd.read_csv(DATA_DIR / nombre)
        for nombre in ARCHIVOS_CSV
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
