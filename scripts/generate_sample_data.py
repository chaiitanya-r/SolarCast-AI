"""
Generate a synthetic SolarIridescenceDataset.csv for pipeline testing.

Run once if the real Khulna dataset is not yet available:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "SolarIridescenceDataset.csv"


def main() -> None:
    rows = []
    for year in range(2014, 2023):
        for month in range(1, 13):
            days_in_month = 30 if month != 2 else 28
            for day in range(1, days_in_month + 1):
                for hour in range(24):
                    temp = 20 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
                    if 6 <= hour <= 19:
                        irr = max(0, 800 * np.sin(np.pi * (hour - 6) / 13) + np.random.normal(0, 50))
                    else:
                        irr = max(0, np.random.normal(5, 3))
                    rows.append(
                        {
                            "Year": year,
                            "Month": month,
                            "Day": day,
                            "Hour": hour,
                            "Temperature": round(temp, 2),
                            "Irradiance": round(irr, 2),
                        }
                    )
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
