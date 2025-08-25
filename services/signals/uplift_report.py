"""Weekly uplift report based on moving average flags."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import pandas as pd

from ..shared.db import get_session
from ..shared.models import Outcome, SignalFeatures, Signal


METRIC_FLAGS = ["price_gt_sma50", "price_gt_ema20", "ema9_gt_ema20", "all_three_true"]


def fetch_week_data(week_end: date) -> pd.DataFrame:
    week_start = week_end - timedelta(days=7)
    with get_session() as session:
        rows = (
            session.query(Outcome, SignalFeatures)
            .join(Signal, Outcome.signal_id == Signal.id)
            .join(SignalFeatures, Outcome.signal_id == SignalFeatures.signal_id)
            .filter(Signal.created_at >= week_start, Signal.created_at < week_end)
            .all()
        )
    data = []
    for out, feat in rows:
        d = {"price_gt_sma50": feat.price_gt_sma50,
             "price_gt_ema20": feat.price_gt_ema20,
             "ema9_gt_ema20": feat.ema9_gt_ema20,
             "r_multiple": out.r_multiple,
             "label": out.label}
        data.append(d)
    return pd.DataFrame(data)


def compute_metrics(df: pd.DataFrame, flag: str) -> pd.Series:
    grp = df.groupby(flag)
    rows = []
    for val, g in grp:
        wins = g[g.r_multiple > 0]
        losses = g[g.r_multiple <= 0]
        profit_factor = wins.r_multiple.sum() / abs(losses.r_multiple.sum()) if len(losses) else float("inf")
        row = {
            flag: val,
            "n": len(g),
            "hit_rate": g.label.mean() if len(g) else 0,
            "profit_factor": profit_factor,
            "exp_R": g.r_multiple.mean() if len(g) else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def generate_weekly_report(week_end: date | None = None) -> pd.DataFrame:
    week_end = week_end or date.today()
    df = fetch_week_data(week_end)
    if df.empty:
        return pd.DataFrame()
    df["all_three_true"] = df.price_gt_sma50 & df.price_gt_ema20 & df.ema9_gt_ema20
    tables = [compute_metrics(df, flag) for flag in METRIC_FLAGS]
    result = pd.concat(tables, keys=METRIC_FLAGS, names=["flag", "row"])
    result = result.reset_index(level=0).drop(columns=["row"])
    result["week_end"] = week_end
    return result


if __name__ == "__main__":
    print(generate_weekly_report())
