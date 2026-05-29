
import pandas as pd

def compute_time_day_summaries(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    activity_col: str = "activity",
    steps_col: str = "steps",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returnerer:
    - daily: minutter pr. aktivitet pr. dag + wear-metrics + total_steps
    - hourly: minutter pr. aktivitet pr. time + wear-metrics + total_steps

    Bruger ALLE aktivitetskategorier, der findes i input-data.
    Antager 1 række = 1 sekund (som i jeres aktivitetsfiler).
    """
    data = df.copy()

    # Sikr datetime-index
    if datetime_col in data.columns:
        data = data.set_index(datetime_col)

    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.isna()].sort_index()

    if activity_col not in data.columns:
        raise ValueError(f"Mangler kolonnen '{activity_col}'")

    # Dynamiske aktivitetskategorier fra data
    data[activity_col] = data[activity_col].astype(str)
    categories = sorted(data[activity_col].dropna().unique().tolist())

    def _summarize(freq: str) -> pd.DataFrame:
        grp = data.index.floor(freq)

        # Antal sekunder pr. aktivitet i hver periode (crosstab giver counts)
        counts = pd.crosstab(grp, data[activity_col]).reindex(columns=categories, fill_value=0)

        # Konverter sekunder -> minutter
        out = counts.astype("float64") / 60.0

        # Observeret tid i perioden (kun tid der faktisk findes i data)
        out["observed_minutes"] = data.groupby(grp).size().astype("float64") / 60.0

        # Non-wear i minutter (0 hvis non-wear ikke findes i perioden/data)
        if "non-wear" in out.columns:
            out["non-wear_minutes"] = out["non-wear"]
        else:
            out["non-wear_minutes"] = 0.0

        # Reel wear-time: observeret tid minus non-wear
        out["wear_minutes"] = (out["observed_minutes"] - out["non-wear_minutes"]).clip(lower=0)

        # Wear-andel af observeret tid
        out["wear_pct_of_observed"] = 0.0
        mask = out["observed_minutes"] > 0
        out.loc[mask, "wear_pct_of_observed"] = (
            100.0 * out.loc[mask, "wear_minutes"] / out.loc[mask, "observed_minutes"]
        )

        # Steps med hvis kolonnen findes
        if steps_col in data.columns:
            out["total_steps"] = data.groupby(grp)[steps_col].sum(min_count=1).fillna(0)

        return out

    daily = _summarize("D")
    hourly = _summarize("h")

    daily.index.name = "date"
    hourly.index.name = "datetime_hour"

    return daily, hourly