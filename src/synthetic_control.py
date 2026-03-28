import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression


def fit_ols_counterfactual(
    prices_norm: pd.DataFrame,
    target: str,
    donor: str,
    pre_end: str,
) -> tuple[LinearRegression, pd.Series]:
    # Обучает OLS на pre-period: target ~ donor.
    # Возвращает (модель, fitted_values_full_period).
    pre = prices_norm.loc[:pre_end, [target, donor]].dropna()
    X_pre = pre[[donor]].values
    y_pre = pre[target].values

    model = LinearRegression().fit(X_pre, y_pre)

    # Контрфактуал на всём периоде
    X_full = prices_norm[[donor]].dropna().values
    idx_full = prices_norm.dropna(subset=[donor]).index
    counterfactual = pd.Series(model.predict(X_full), index=idx_full, name="counterfactual")

    return model, counterfactual


def bootstrap_ci(
    prices_norm: pd.DataFrame,
    target: str,
    donor: str,
    pre_end: str,
    post_start: str,
    n_boot: int = 500,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Бутстрэп для оценки CI контрфактуала.
    rng = np.random.default_rng(seed)
    pre_data = prices_norm.loc[:pre_end, [target, donor]].dropna()
    post_data = prices_norm.loc[post_start:, [target, donor]].dropna()

    boot_preds = np.zeros((n_boot, len(post_data)))

    for i in range(n_boot):
        idx = rng.integers(0, len(pre_data), size=len(pre_data))
        sample = pre_data.iloc[idx]
        m = LinearRegression().fit(sample[[donor]].values, sample[target].values)
        boot_preds[i] = m.predict(post_data[[donor]].values)

    alpha = (1 - ci_level) / 2
    lower = pd.Series(np.quantile(boot_preds, alpha,   axis=0), index=post_data.index)
    upper = pd.Series(np.quantile(boot_preds, 1-alpha, axis=0), index=post_data.index)
    mean  = pd.Series(np.mean(boot_preds, axis=0),              index=post_data.index)

    return mean, lower, upper


def compute_gap_effect(
    prices_norm: pd.DataFrame,
    target: str,
    counterfactual: pd.Series,
    post_start: str,
) -> dict:
    # Считает эффект контрфактуала.
    actual = prices_norm.loc[post_start:, target].dropna()
    cf     = counterfactual.loc[post_start:].reindex(actual.index)
    gap    = actual - cf

    return {
        "avg_actual":      round(actual.mean(), 2),
        "avg_counterfact": round(cf.mean(), 2),
        "avg_gap":         round(gap.mean(), 2),
        "rel_effect_pct":  round((gap / cf).mean() * 100, 1),
        "cumul_gap":       round(gap.sum(), 2),
    }


def plot_synthetic_control(
    prices_norm: pd.DataFrame,
    target: str,
    counterfactual: pd.Series,
    lower_ci: pd.Series,
    upper_ci: pd.Series,
    event_date: str,
    title: str,
    ax: plt.Axes,
) -> None:
    event = pd.Timestamp(event_date)

    ax.plot(prices_norm.index, prices_norm[target],
            color="black", lw=1.5, label=f"{target} (observed)")
    ax.plot(counterfactual.index, counterfactual,
            color="steelblue", lw=1.5, ls="--", label="Counterfactual (OLS)")
    ax.fill_between(lower_ci.index, lower_ci, upper_ci,
                    alpha=0.2, color="steelblue", label="95% Bootstrap CI")
    ax.axvline(event, color="red", ls=":", lw=2, label="Event date")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Normalised Price (100 = Jan 2010)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))