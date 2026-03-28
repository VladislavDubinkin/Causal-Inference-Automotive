import pandas as pd
import numpy as np

# ── Замени на свой username и repo после пуша ──────────────────────────
GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/VladislavDubinkin/"
    "Causal-Inference-Automotive/main/data/United_Dataset.csv"
)

DIESELGATE_DATE = "2015-09-18"
COVID_DATE      = "2020-02-24"
ROLLING_WINDOW  = 21
ANNUALISE       = np.sqrt(252)


def load_raw(url: str = GITHUB_RAW_URL) -> pd.DataFrame:
    """
    Читает CSV с GitHub, исправляет европейский формат чисел,
    парсит даты, возвращает чистый long-format DataFrame.
    """
    df = pd.read_csv(
        url,
        sep=";",
        decimal=",",
        encoding="utf-8",
        lineterminator="\n",
    )

    df.columns = df.columns.str.strip().str.replace("\r", "", regex=False)
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().str.replace("\r", "", regex=False)

    df["DAX"] = (
        df["DAX"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["Change"] = (
        df["Change"]
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    return df


def compute_volatility(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    vol = (
        df.groupby("Company")["log_return"]
        .apply(lambda s: s.rolling(window).std() * ANNUALISE)
        .reset_index(level=0, drop=True)
    )
    df = df.copy()
    df["volatility"] = vol

    wide = (
        df.dropna(subset=["volatility"])
          .pivot_table(index="Date", columns="Company", values="volatility")
          .sort_index()
    )
    wide.columns.name = None
    return wide


def compute_prices_normalized(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="Date", columns="Company", values="Price").sort_index()
    wide.columns.name = None
    wide_norm = wide.div(wide.iloc[0]) * 100

    # Добавляем DAX нормализованный
    dax = df.drop_duplicates("Date").set_index("Date")["DAX"].sort_index()
    wide_norm["DAX"] = dax / dax.iloc[0] * 100

    return wide_norm


def get_event_windows() -> dict:
    return {
        "dieselgate": {
            "pre":  ("2013-01-01", "2015-09-17"),
            "post": ("2015-09-18", "2016-06-30"),
            "label": "Dieselgate (Sep 2015)",
        },
        "covid": {
            "pre":  ("2017-01-01", "2020-02-21"),
            "post": ("2020-02-24", "2021-06-30"),
            "label": "COVID-19 (Feb 2020)",
        },
    }