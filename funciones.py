import unicodedata
from functools import lru_cache
from typing import Literal

import pandas as pd


TipoOrdenDerrotas = Literal["absoluto", "porcentaje", "ponderado"]


def clasificar_mitad(minuto: object) -> str:
    if pd.isna(minuto):
        return "Desconocido"
    return "Primer Tiempo (1-45)" if minuto <= 45 else "Segundo Tiempo (46+)"


@lru_cache(maxsize=None)
def clasificar_torneo(nombre: str) -> str:
    normalizado = (
        unicodedata.normalize("NFKD", str(nombre))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    if "friendly" in normalizado:
        return "Amistoso"
    if "qualification" in normalizado:
        return "Eliminatorias / Clasificacion"
    if "world cup" in normalizado:
        return "Copa del Mundo (Fase Final)"
    if any(
        torneo in normalizado
        for torneo in (
            "euro",
            "copa america",
            "african cup",
            "asian cup",
            "gold cup",
            "nations cup",
            "championship",
        )
    ):
        return "Torneo Continental"
    return "Otros / Regionales"


def preparar_registros_selecciones(resultados: pd.DataFrame) -> pd.DataFrame:
    columnas = ["seleccion", "rival", "goles_favor", "goles_contra"]
    local = resultados[
        ["home_team", "away_team", "home_score", "away_score"]
    ].set_axis(columnas, axis=1)
    visitante = resultados[
        ["away_team", "home_team", "away_score", "home_score"]
    ].set_axis(columnas, axis=1)
    registros = pd.concat([local, visitante], ignore_index=True)
    registros["derrota"] = (
        registros["goles_contra"] > registros["goles_favor"]
    ).astype(int)
    return registros


def ranking_derrotas(
    registros: pd.DataFrame,
    seleccion: str,
    orden: TipoOrdenDerrotas,
) -> pd.DataFrame:
    if orden not in {"absoluto", "porcentaje", "ponderado"}:
        raise ValueError(f"Orden de ranking no válido: {orden!r}")

    columnas_orden = {
        "absoluto": "derrotas",
        "porcentaje": "porcentaje_derrotas",
        "ponderado": "score_ponderado",
    }
    sub = registros[registros["seleccion"] == seleccion]
    resumen = (
        sub.groupby("rival")["derrota"]
        .agg(derrotas="sum", total_partidos="count")
        .reset_index()
    )
    if resumen.empty:
        return resumen

    resumen["porcentaje_derrotas"] = (
        resumen["derrotas"] / resumen["total_partidos"] * 100
    ).round(1)
    if orden == "ponderado":
        max_derrotas = max(resumen["derrotas"].max(), 1)
        max_porcentaje = max(resumen["porcentaje_derrotas"].max(), 1)
        resumen["score_ponderado"] = (
            0.5 * resumen["derrotas"] / max_derrotas
            + 0.5 * resumen["porcentaje_derrotas"] / max_porcentaje
        ).mul(100).round(1)
    return resumen.sort_values(
        columnas_orden[orden], ascending=False, ignore_index=True
    ).head(15)


def obtener_bestia_negra(
    registros: pd.DataFrame, seleccion: str
) -> pd.Series:
    ranking = ranking_derrotas(registros, seleccion, "absoluto")
    if ranking.empty:
        raise ValueError(f"No hay partidos registrados para {seleccion!r}.")
    return ranking.iloc[0]
