import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)   # Fixed seed — keeps data consistent across all steps

# ── Grid layout ──────────────────────────────────────────────
# 20 smart meters spread across three consumer types
METER_CONFIG = [
    ("residential",  12),   # houses / apartments
    ("commercial",    5),   # offices / shops
    ("industrial",    3),   # factories / warehouses
]

# ── Time range: July 2024, hourly readings ────────────────────
START    = datetime(2024, 7, 1, 0, 0)
N_HOURS  = 31 * 24          # 744 hours  →  14 880 rows total (20 meters × 744)
timestamps = [START + timedelta(hours=h) for h in range(N_HOURS)]

# ── Base consumption (kWh per hour at "normal" conditions) ────
BASE_KWH = {"residential": 1.5, "commercial": 8.0, "industrial": 25.0}

rows = []

for meter_type, count in METER_CONFIG:
    base = BASE_KWH[meter_type]

    for idx in range(count):
        # Give each meter a slight personality so readings aren't identical
        meter_scale = np.random.uniform(0.8, 1.2)

        for ts in timestamps:
            hour = ts.hour
            dow  = ts.weekday()          # 0 = Monday … 6 = Sunday
            dom  = ts.day

            # ── Synthetic temperature (°C) ────────────────────
            # July summer: peaks ~3 pm (hour 15), troughs ~5 am (hour 5)
            daily_swing  = 8 * np.sin((hour - 5) * np.pi / 14)
            week_wave    = 2 * np.sin((dom / 31) * 2 * np.pi)   # slight monthly drift
            temperature  = 26 + daily_swing + week_wave + np.random.normal(0, 1.2)
            temperature  = float(np.clip(temperature, 16, 41))

            # ── Consumption model ─────────────────────────────
            if meter_type == "residential":
                # Two peaks: morning commute prep (7–9 am) & evening home (6–10 pm)
                morning  = 0.7 * np.exp(-((hour -  8) ** 2) / 6)
                evening  = 1.0 * np.exp(-((hour - 20) ** 2) / 8)
                base_curve = 0.35 + morning + evening
                ac_boost   = max(0.0, (temperature - 23) * 0.09)   # AC kicks in above 23°C
                wknd       = 1.25 if dow >= 5 else 1.0              # people home more
                usage      = base * meter_scale * (base_curve + ac_boost) * wknd

            elif meter_type == "commercial":
                # Business hours Mon–Fri; skeleton load on weekends
                if dow < 5:
                    open_  = 1 / (1 + np.exp(-(hour -  8)))        # sigmoid open  ~8 am
                    close_ = 1 / (1 + np.exp(-(hour - 18)))        # sigmoid close ~6 pm
                    base_curve = 0.25 + 1.5 * (open_ - close_)
                else:
                    base_curve = 0.15
                ac_boost = max(0.0, (temperature - 23) * 0.18)
                usage    = base * meter_scale * (base_curve + ac_boost)

            else:  # industrial
                # Fairly flat; minor dip overnight and slight weekend reduction
                shift_on  = 1.0 if 6 <= hour <= 22 else 0.75
                wknd      = 0.85 if dow >= 5 else 1.0
                usage     = base * meter_scale * shift_on * wknd

            # Small random measurement noise (±5 %)
            usage += np.random.normal(0, usage * 0.05)
            usage  = max(0.0, usage)

            rows.append({
                "timestamp":    ts.strftime("%Y-%m-%d %H:%M"),
                "meter_id":     f"{meter_type}_{idx+1:02d}",
                "meter_type":   meter_type,
                "hour":         hour,
                "day_of_week":  dow,          # 0=Mon … 6=Sun
                "day_of_month": dom,
                "week_of_month":(dom - 1) // 7 + 1,
                "is_weekend":   int(dow >= 5),
                "temperature_c": round(temperature, 1),
                "energy_kwh":   round(usage, 4),
            })

df = pd.DataFrame(rows)

from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parent / "smart_grid_july.csv"

df.to_csv(OUTPUT_FILE, index=False)

# ── Summary ───────────────────────────────────────────────────
print("=" * 52)
print("  smart_grid_july.csv  generated successfully")
print("=" * 52)
print(f"  Total rows      : {len(df):,}")
print(f"  Columns         : {list(df.columns)}")
print(f"  Date range      : {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
print(f"  Meters          : {df['meter_id'].nunique()} unique")
print()
print(df.groupby("meter_type")["energy_kwh"].describe().round(3))
print()
print("  Open smart_grid_july.csv in Excel / any spreadsheet viewer")
print("  to explore the raw readings before starting the lab.\n")
