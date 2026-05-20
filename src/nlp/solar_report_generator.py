"""
Rule-based NLP solar conditions report generator (offline, no LLM).

Given sensor/prediction values, produces natural-language solar reports.
"""

from __future__ import annotations

import pandas as pd

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, timer


def _ghi_class(ghi: float) -> str:
    if ghi < 200:
        return "Low"
    if ghi <= 600:
        return "Medium"
    return "High"


def _temp_class(temp: float) -> str:
    if temp < 20:
        return "Cool"
    if temp <= 32:
        return "Warm"
    return "Hot"


def _season(month: int | None) -> str:
    if month is None:
        return "current"
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _time_of_day(hour: int | None) -> str:
    if hour is None:
        return "daytime"
    if hour < 10:
        return "early morning"
    if hour < 13:
        return "midday"
    if hour < 17:
        return "afternoon"
    return "evening"


def generate_solar_report(
    ghi: float,
    temperature: float,
    humidity: float | None = None,
    hour: int | None = None,
    month: int | None = None,
) -> str:
    """
    Given numeric sensor/prediction values, return a natural-language solar report.
    """
    ghi_level = _ghi_class(ghi)
    temp_level = _temp_class(temperature)
    season = _season(month)
    tod = _time_of_day(hour)

    if ghi_level == "High":
        gen_sentence = (
            f"High solar generation conditions are expected during {tod} hours "
            f"under clear atmospheric conditions in {season}."
        )
    elif ghi_level == "Medium":
        gen_sentence = (
            f"Moderate solar generation is forecast for {tod} hours during {season} "
            f"with partially favorable irradiance."
        )
    else:
        gen_sentence = (
            f"Low solar generation is forecast for {tod} hours. "
            f"Overcast {season} conditions suggest minimal energy yield."
        )

    hum_note = ""
    if humidity is not None:
        if humidity < 40:
            hum_note = " Low humidity indicates strong irradiance potential."
        elif humidity > 70:
            hum_note = " High humidity may attenuate direct normal irradiance."
        else:
            hum_note = " Moderate humidity suggests typical atmospheric transmission."

    temp_sentence = (
        f"A temperature of {temperature:.0f}°C ({temp_level})"
        + (hum_note if hum_note else " supports the irradiance outlook.")
        + (
            " This is optimal for peak photovoltaic output."
            if ghi_level == "High" and temp_level in ("Warm", "Hot")
            else " Cloud diffusion may reduce direct normal irradiance significantly."
            if ghi_level == "Low"
            else " Conditions are suitable for steady photovoltaic production."
        )
    )

    return f"{gen_sentence} {temp_sentence}"


def batch_generate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Apply generate_solar_report to every row; append nlp_report column."""
    out = df.copy()

    def _row_report(row: pd.Series) -> str:
        return generate_solar_report(
            ghi=row.get("Irradiance", row.get("ghi", 0)),
            temperature=row.get("Temperature", row.get("temperature", 25)),
            humidity=row.get("humidity", None) if "humidity" in row else None,
            hour=int(row["Hour"]) if "Hour" in row and pd.notna(row["Hour"]) else None,
            month=int(row["Month"]) if "Month" in row and pd.notna(row["Month"]) else None,
        )

    out["nlp_report"] = out.apply(_row_report, axis=1)
    return out


def save_sample_reports(df: pd.DataFrame, n: int = 10) -> None:
    """Save n sample NLP reports to results/reports/sample_nlp_reports.txt."""
    ensure_dirs()
    path = get_project_root() / "results" / "reports" / "sample_nlp_reports.txt"
    samples = df.head(n)
    lines = []
    for i, row in samples.iterrows():
        lines.append(f"--- Sample {i} ---")
        lines.append(row.get("nlp_report", generate_solar_report(row.get("Irradiance", 0), row.get("Temperature", 25))))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {n} sample reports to {path}")


@timer
def run_nlp() -> pd.DataFrame:
    df = load_processed("engineered_solar_data")
    df = batch_generate_report(df)
    save_sample_reports(df, n=10)
    print(df[["Irradiance", "Temperature", "nlp_report"]].head(3))
    return df


def main() -> pd.DataFrame:
    return run_nlp()


if __name__ == "__main__":
    main()
