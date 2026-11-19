import pandas as pd

from graficos import generar_graficos


def main(dataframes: dict[str, pd.DataFrame] | None = None) -> None:
    generar_graficos(dataframes)


if __name__ == "__main__":
    main()
