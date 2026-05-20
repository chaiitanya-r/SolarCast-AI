"""
Rule-based NLP solar conditions report generator (offline, no LLM).

Given sensor/prediction values, produces natural-language solar reports.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, save_plot, timer


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
    print(f"  Saved: sample_nlp_reports.txt")


@timer
def run_nlp() -> pd.DataFrame:
    df = load_processed("engineered_solar_data")
    df = batch_generate_report(df)
    save_sample_reports(df, n=10)
    try:
        plt.style.use("seaborn-v0_8")
        sample_size = min(500, len(df))
        sampled = batch_generate_report(df.sample(n=sample_size, random_state=42))
        labels = []
        for text in sampled["nlp_report"]:
            t = str(text).lower()
            if "high solar generation" in t:
                labels.append("High")
            elif "moderate solar generation" in t:
                labels.append("Medium")
            else:
                labels.append("Low")
        counts = pd.Series(labels).value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.tab10([0, 1, 2]),
        )
        ax.set_title("Generated Report Class Distribution (Sample)")
        plt.tight_layout()
        save_plot(fig, "nlp_report_class_distribution.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate NLP class distribution plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        stopwords = {
            "the", "a", "is", "are", "and", "of", "for", "in", "to", "under",
            "with", "this", "be", "may", "will",
        }
        words: list[str] = []
        for text in df["nlp_report"].astype(str):
            for w in text.lower().replace(".", " ").replace(",", " ").split():
                if w and w not in stopwords:
                    words.append(w)
        freq = Counter(words)
        top = freq.most_common(20)
        if top:
            labels = [w for w, _ in top][::-1]
            values = [c for _, c in top][::-1]
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(labels, values, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(labels))))
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set_title("Top 20 Word Frequencies in Generated Reports")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Word")
            plt.tight_layout()
            save_plot(fig, "nlp_word_frequency.png")
            plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate NLP word frequency chart: {exc}")
    print(f"  Output: {len(df):,} rows with nlp_report column")
    return df


def main() -> pd.DataFrame:
    return run_nlp()


if __name__ == "__main__":
    main()
