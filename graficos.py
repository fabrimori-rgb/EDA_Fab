from pathlib import Path
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from funciones import (
    clasificar_torneo,
    preparar_registros_selecciones,
    ranking_derrotas,
)
from variables import cargar_datos, preparar_goles_mundial


OUTPUT_DIR = Path(__file__).resolve().parent / "graficos"
TRAMOS = ["0-15", "16-30", "31-45+", "46-60", "61-75", "76-90+", "Prórroga"]


def guardar_figura(fig: plt.Figure, nombre: str, figuras: list[plt.Figure]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = OUTPUT_DIR / f"{nombre}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    figuras.append(fig)
    print(f"Gráfico guardado: {ruta}")


def preparar_goles(df_goles: pd.DataFrame) -> pd.DataFrame:
    goles = df_goles.copy()
    minuto_base = (
        goles["minute"].astype("string").str.split("+").str[0].str.strip()
    )
    goles["minuto_num"] = pd.to_numeric(minuto_base, errors="coerce")
    goles["tramo"] = np.select(
        [
            goles["minuto_num"].between(0, 15),
            goles["minuto_num"].between(16, 30),
            goles["minuto_num"].between(31, 45),
            goles["minuto_num"].between(46, 60),
            goles["minuto_num"].between(61, 75),
            goles["minuto_num"].between(76, 90),
            goles["minuto_num"].gt(90),
            goles["minuto_num"].isna(),
        ],
        TRAMOS + ["Sin dato"],
        default="Sin dato",
    )
    no_interpretados = minuto_base.notna() & goles["minuto_num"].isna()
    print(f"Minutos no interpretados: {no_interpretados.sum()}")
    return goles


def graficar_goles_por_tipo(
    resultados: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    partidos = resultados.copy()
    partidos["total_goles"] = partidos["home_score"] + partidos["away_score"]
    partidos["tipo_partido"] = np.where(
        partidos["tournament"].eq("Friendly"), "Amistoso", "Oficial"
    )
    promedios = (
        partidos.groupby("tipo_partido")["total_goles"].mean().reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(
        data=promedios,
        x="tipo_partido",
        y="total_goles",
        hue="tipo_partido",
        ax=axes[0],
        palette="colorblind",
        legend=False,
    )
    axes[0].set_title("Promedio de Goles por Tipo de Partido", fontweight="bold")
    axes[0].set_xlabel("Tipo de Partido")
    axes[0].set_ylabel("Promedio de Goles")

    sns.boxplot(
        data=partidos,
        x="total_goles",
        y="tipo_partido",
        hue="tipo_partido",
        ax=axes[1],
        palette="magma",
        legend=False,
    )
    axes[1].set_title("Distribución de Goles", fontweight="bold")
    axes[1].set_xlabel("Total de Goles")
    axes[1].set_ylabel("Tipo de Partido")
    fig.tight_layout()
    guardar_figura(fig, "01_goles_por_tipo_de_partido", figuras)


def graficar_goles_por_tramo(
    goles: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    conteo_tramos = (
        goles["tramo"].value_counts().reindex(TRAMOS + ["Sin dato"], fill_value=0)
    )
    df_tramos = pd.DataFrame(
        {"Tramo": conteo_tramos.index, "Cantidad": conteo_tramos.values}
    )

    minuto = goles["minuto_num"]
    mitades = np.select(
        [minuto.between(1, 45), minuto.ge(46)],
        ["Primer Tiempo (1-45)", "Segundo Tiempo (46+)"],
        default="Desconocido",
    )
    orden_mitades = [
        "Primer Tiempo (1-45)",
        "Segundo Tiempo (46+)",
        "Desconocido",
    ]
    conteo_mitades = (
        pd.Series(mitades)
        .value_counts()
        .reindex(orden_mitades, fill_value=0)
    )
    df_mitades = pd.DataFrame(
        {"Mitad": conteo_mitades.index, "Cantidad": conteo_mitades.values}
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=df_tramos,
        x="Tramo",
        y="Cantidad",
        hue="Tramo",
        palette="Greens_r",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Goles por Tramo de 15 Minutos", fontweight="bold")
    axes[0].set_xlabel("Tramo del Partido")
    axes[0].set_ylabel("Cantidad de Goles")
    axes[0].tick_params(axis="x", rotation=30)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.0f", padding=3)

    sns.barplot(
        data=df_mitades,
        x="Mitad",
        y="Cantidad",
        hue="Mitad",
        palette="Set2",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Goles por Mitad de Partido", fontweight="bold")
    axes[1].set_xlabel("Fase del partido")
    axes[1].set_ylabel("Cantidad de Goles")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.0f", padding=3)

    fig.tight_layout()
    guardar_figura(fig, "02_goles_por_tramos_y_mitades", figuras)


def graficar_tipos_de_gol(
    goles: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    total = len(goles)
    autogoles = int(goles["own_goal"].sum())
    penalties = int(goles["penalty"].sum())
    normales = total - autogoles - penalties
    df_tipos = pd.DataFrame(
        {
            "Tipología": ["Jugada Colectiva", "Penalties", "Autogol"],
            "Porcentaje": [
                normales / total * 100,
                penalties / total * 100,
                autogoles / total * 100,
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=df_tipos,
        x="Porcentaje",
        y="Tipología",
        hue="Tipología",
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=4, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_title("Impacto de la Tipología de Goles", fontweight="bold")
    ax.set_xlabel("Porcentaje (%)")
    ax.set_ylabel("")
    fig.tight_layout()
    guardar_figura(fig, "03_tipos_de_gol", figuras)


def graficar_penalties_y_autogoles(
    goles: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    rows = []
    for columna, tipo in [("penalty", "Penalties"), ("own_goal", "Autogoles")]:
        porcentajes = (
            goles.loc[goles[columna].astype(bool), "tramo"]
            .value_counts(normalize=True)
            .reindex(TRAMOS, fill_value=0)
            .mul(100)
        )
        rows.extend(
            {"Tramo": tramo, "Tipo": tipo, "Porcentaje": porcentaje}
            for tramo, porcentaje in porcentajes.items()
        )
    df_tramos_pct = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(
        data=df_tramos_pct,
        x="Tramo",
        y="Porcentaje",
        hue="Tipo",
        style="Tipo",
        s=120,
        palette="Set1",
        ax=ax,
    )
    ax.set_title(
        "Distribución por Tramos: Penalties vs Autogoles",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Tramo del Partido")
    ax.set_ylabel("Porcentaje del Total (%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    guardar_figura(fig, "04_penalties_y_autogoles_por_tramo", figuras)


def graficar_factor_campo(
    resultados: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    partidos = resultados.copy()
    partidos["home_win"] = partidos["home_score"] > partidos["away_score"]
    porcentajes = partidos.groupby("neutral")["home_win"].mean() * 100
    valores = [round(porcentajes.get(False, 0), 2), round(porcentajes.get(True, 0), 2)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(["En casa", "Campo neutral"], valores, color=["#3498db", "#3E053F"])
    ax.set_xlabel("Porcentaje de victorias (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Peso del Factor Campo")
    for i, valor in enumerate(valores):
        ax.text(valor + 1, i, f"{valor}%", va="center", fontweight="bold")
    fig.tight_layout()
    guardar_figura(fig, "05_factor_campo", figuras)


def graficar_victorias_por_torneo(
    resultados: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    partidos = resultados.copy()
    partidos["tipo_torneo"] = partidos["tournament"].apply(clasificar_torneo)
    partidos["ganador"] = None
    partidos.loc[
        partidos["home_score"] > partidos["away_score"], "ganador"
    ] = partidos["home_team"]
    partidos.loc[
        partidos["away_score"] > partidos["home_score"], "ganador"
    ] = partidos["away_team"]
    victorias = (
        partidos.dropna(subset=["ganador"])
        .groupby(["tipo_torneo", "ganador"])
        .size()
        .reset_index(name="total_victorias")
    )
    victorias = victorias[victorias["tipo_torneo"] != "Amistoso"]
    top_5 = (
        victorias.sort_values(
            ["tipo_torneo", "total_victorias"], ascending=[True, False]
        )
        .groupby("tipo_torneo")
        .head(5)
    )

    tipologias = top_5["tipo_torneo"].unique()
    filas = max(1, (len(tipologias) + 1) // 2)
    fig, axes = plt.subplots(filas, 2, figsize=(14, 5 * filas), squeeze=False)
    axes = axes.flatten()
    for i, tipo in enumerate(tipologias):
        ax = axes[i]
        datos_tipo = top_5[top_5["tipo_torneo"] == tipo].iloc[::-1]
        barras = ax.barh(
            datos_tipo["ganador"], datos_tipo["total_victorias"], color="#3498db"
        )
        ax.bar_label(barras, padding=4, fontweight="bold", fontsize=9)
        ax.set_xlim(0, datos_tipo["total_victorias"].max() * 1.1)
        ax.set_title(f"Top 5 - {tipo}", fontweight="bold")
        ax.set_xlabel("Total de victorias")
        ax.grid(axis="x", linestyle="--", alpha=0.7)
    for ax in axes[len(tipologias):]:
        ax.set_visible(False)
    fig.tight_layout()
    guardar_figura(fig, "06_victorias_por_tipo_de_torneo", figuras)


def graficar_ranking_derrotas(
    registros: pd.DataFrame,
    figuras: list[plt.Figure],
    orden: str,
    titulo: str,
    xlabel: str,
    nombre_archivo: str,
    palette: str,
) -> None:
    resumen = ranking_derrotas(registros, "France", orden)
    if resumen.empty:
        print("No hay datos para France; se omite un gráfico de derrotas.")
        return
    medida = {
        "porcentaje": "porcentaje_derrotas",
        "absoluto": "derrotas",
        "ponderado": "score_ponderado",
    }[orden]
    orden_rivales = resumen["rival"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=resumen,
        x=medida,
        y="rival",
        order=orden_rivales,
        hue="rival",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rival")
    fig.tight_layout()
    guardar_figura(fig, nombre_archivo, figuras)


def graficar_tandas(
    tandas: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    validas = tandas.dropna(subset=["first_shooter"]).copy()
    if validas.empty:
        print("No hay tandas con first_shooter; se omite el gráfico.")
        return
    validas["gana_primero"] = validas["first_shooter"] == validas["winner"]
    victorias_primero = int(validas["gana_primero"].sum())
    total = len(validas)
    resumen = pd.DataFrame(
        {
            "Turno de Lanzamiento": ["Tiran Primero", "Tiran Segundo"],
            "Porcentaje": [
                victorias_primero / total * 100,
                (total - victorias_primero) / total * 100,
            ],
            "Cantidad": [victorias_primero, total - victorias_primero],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=resumen,
        x="Porcentaje",
        y="Turno de Lanzamiento",
        hue="Turno de Lanzamiento",
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=4, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_title(
        f"Tandas de Penaltis Post-Partido: ¿Ventaja por el Orden? (n={total})",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Porcentaje de Victorias en la Tanda (%)")
    ax.set_ylabel("")
    fig.tight_layout()
    guardar_figura(fig, "10_orden_de_lanzamiento_en_tandas", figuras)


def reparar_nombre_scorer(nombre: str) -> str:
    if any(marca in nombre for marca in ("Ã", "Â")) and "\ufffd" not in nombre:
        return nombre.encode("latin-1").decode("utf-8")
    return nombre


def graficar_pichichi_mundial(
    df_mundial: pd.DataFrame, figuras: list[plt.Figure]
) -> None:
    goles_tempraneros = df_mundial.loc[
        df_mundial["minuto_num"].between(1, 15)
    ]
    pichichi = (
        goles_tempraneros.groupby("scorer")
        .size()
        .reset_index(name="goles_primeros_15")
        .sort_values("goles_primeros_15", ascending=False)
    )
    top_10 = pichichi.head(10).sort_values(
        "goles_primeros_15", ascending=True
    )
    if top_10.empty:
        print("No hay goles de Mundial en los primeros 15 minutos.")
        return

    top_10["scorer"] = top_10["scorer"].map(reparar_nombre_scorer)
    orden = top_10["scorer"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=top_10,
        x="goles_primeros_15",
        y="scorer",
        hue="scorer",
        order=orden,
        hue_order=orden,
        palette="Greens_r",
        dodge=False,
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(
            container, fmt="%d goles", padding=4, fontweight="bold"
        )
    ax.set_xlim(0, top_10["goles_primeros_15"].max() + 1.5)
    ax.set_title(
        "El Pichichi más rápido: Máximos goleadores (Minutos 1-15 en Mundiales)",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Cantidad de Goles en los primeros 15 minutos")
    ax.set_ylabel("")
    ax.invert_yaxis()
    sns.despine(ax=ax)
    fig.tight_layout()
    guardar_figura(fig, "11_top_10_pichichi_tempranero_mundial", figuras)


def generar_graficos(
    dataframes: dict[str, pd.DataFrame] | None = None,
) -> None:
    if dataframes is None:
        dataframes = cargar_datos()
    print("DataFrames cargados:")
    for nombre, dataframe in dataframes.items():
        print(f"- {nombre}: {dataframe.shape[0]} filas, {dataframe.shape[1]} columnas")

    sns.set_theme(style="whitegrid", palette="bright")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figuras: list[plt.Figure] = []
    resultados = dataframes["resultados_limpio"]
    goles = preparar_goles(dataframes["marcadores_limpio"])

    graficar_goles_por_tipo(resultados, figuras)
    graficar_goles_por_tramo(goles, figuras)
    graficar_tipos_de_gol(goles, figuras)
    graficar_penalties_y_autogoles(goles, figuras)
    graficar_factor_campo(resultados, figuras)
    graficar_victorias_por_torneo(resultados, figuras)

    registros = preparar_registros_selecciones(resultados)
    graficar_ranking_derrotas(
        registros,
        figuras,
        "porcentaje",
        "Ranking de Derrotas (%) - France",
        "% de Derrotas",
        "07_derrotas_porcentaje_france",
        "Reds_r",
    )
    graficar_ranking_derrotas(
        registros,
        figuras,
        "absoluto",
        "Ranking Absoluto de Derrotas - France",
        "Cantidad de Derrotas",
        "08_derrotas_absolutas_france",
        "Blues_r",
    )
    graficar_ranking_derrotas(
        registros,
        figuras,
        "ponderado",
        "Ranking Ponderado de Bestia Negra (Volumen + %) - France",
        "Índice Ponderado de Severidad",
        "09_bestia_negra_ponderada_france",
        "Purples_r",
    )
    graficar_tandas(dataframes["penalties_limpio"], figuras)
    df_mundial = preparar_goles_mundial(dataframes)
    graficar_pichichi_mundial(df_mundial, figuras)

    print(f"\nSe guardaron {len(figuras)} gráficos en: {OUTPUT_DIR}")
    print("Se abrirán las ventanas de los gráficos; ciérralas para terminar.")
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    generar_graficos()
