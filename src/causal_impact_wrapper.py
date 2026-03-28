import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from causalimpact import CausalImpact


def run_causal_impact(
    vol_wide: pd.DataFrame,
    target: str,
    controls: list[str],
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
) -> CausalImpact:
    cols = [target] + controls
    data = vol_wide[cols].loc[pre_period[0]:post_period[1]].dropna()

    ci = CausalImpact(data, list(pre_period), list(post_period),
                      model_args={"nseasons": 5, "season_duration": 1})
    ci.run()
    return ci


def extract_summary_metrics(ci: CausalImpact, label: str) -> dict:
    # Извлекаем ключевые метрики из объекта CausalImpact.
    s = ci.summary_data
    return {
        "label":           label,
        "actual_mean":     round(s.loc["average", "actual"],     4),
        "pred_mean":       round(s.loc["average", "predicted"],  4),
        "abs_effect":      round(s.loc["average", "abs_effect"], 4),
        "rel_effect_pct":  round(s.loc["average", "rel_effect"] * 100, 1),
        "p_value":         round(ci.p_value, 4),
        "ci_lower":        round(s.loc["average", "abs_effect_lower"], 4),
        "ci_upper":        round(s.loc["average", "abs_effect_upper"], 4),
    }


def placebo_test(
    vol_wide: pd.DataFrame,
    true_target: str,
    controls: list[str],
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
) -> tuple[float, list[float]]:
    # Permutation test: запускает CausalImpact для каждой контрольной
    # единицы как если бы она была treated.
    # Возвращает (real_effect, [placebo_effects]).

    ci_real = run_causal_impact(vol_wide, true_target, controls,
                                pre_period, post_period)
    real_effect = ci_real.summary_data.loc["average", "abs_effect"]

    placebo_effects = []
    for placebo_unit in controls:
        remaining = [c for c in controls if c != placebo_unit]
        if len(remaining) == 0:
            continue
        try:
            ci_p = run_causal_impact(vol_wide, placebo_unit, remaining,
                                     pre_period, post_period)
            placebo_effects.append(
                ci_p.summary_data.loc["average", "abs_effect"]
            )
        except Exception:
            pass

    return real_effect, placebo_effects


def plot_causal_impact(
    ci: CausalImpact,
    event_date: str,
    title: str,
    ax_triplet: list,
) -> None:
    # Рисуем три панели CausalImpact на переданных осях.
    # ax_triplet = [ax_observed, ax_pointwise, ax_cumulative]
    inf = ci.inferences
    event = pd.Timestamp(event_date)

    # Панель 1: факт vs контрфактуал
    ax = ax_triplet[0]
    ax.plot(inf.index, inf["response"],       color="black",    lw=1.5, label="Observed")
    ax.plot(inf.index, inf["point_pred"],     color="steelblue", lw=1.5,
            ls="--", label="Counterfactual")
    ax.fill_between(inf.index,
                    inf["point_pred_lower"],
                    inf["point_pred_upper"],
                    alpha=0.2, color="steelblue")
    ax.axvline(event, color="red", ls=":", lw=2)
    ax.set_title(f"{title} — Observed vs Counterfactual", fontweight="bold")
    ax.set_ylabel("Ann. Volatility")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Панель 2: pointwise effect
    ax = ax_triplet[1]
    ax.plot(inf.index, inf["point_effect"], color="darkred", lw=1.2)
    ax.fill_between(inf.index,
                    inf["point_effect_lower"],
                    inf["point_effect_upper"],
                    alpha=0.2, color="red")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.axvline(event, color="red", ls=":", lw=2)
    ax.set_title("Pointwise Effect", fontweight="bold")
    ax.set_ylabel("Δ Volatility")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Панель 3: кумулятивный эффект
    ax = ax_triplet[2]
    ax.plot(inf.index, inf["cum_effect"], color="darkgreen", lw=1.5)
    ax.fill_between(inf.index,
                    inf["cum_effect_lower"],
                    inf["cum_effect_upper"],
                    alpha=0.2, color="green")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.axvline(event, color="red", ls=":", lw=2)
    ax.set_title("Cumulative Effect", fontweight="bold")
    ax.set_ylabel("Cumul. Δ Volatility")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))