import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "smart_grid_july.csv"

np.random.seed(42)

# ── 1. Load full dataset ──────────────────────────────────────
df = pd.read_csv(DATA_FILE)
TARGET = "energy_kwh"

# ── 2. Feature engineering ────────────────────────────────────
# Raw features (same as Steps 1 & 2)
FEATURES_RAW = ["hour", "day_of_week", "temperature_c"]

# Engineered features: add domain knowledge about the smart grid
df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)      # cyclical hour encoding
df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)      # captures "wrap-around" at midnight
df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["temp_sq"]      = df["temperature_c"] ** 2                  # AC load grows non-linearly with heat
df["heat_index"]   = df["temperature_c"] * (1 - df["is_weekend"] * 0.2)  # heat × occupancy proxy

# Encode meter type as numeric dummies
df = pd.get_dummies(df, columns=["meter_type"], drop_first=False)
type_cols = [c for c in df.columns if c.startswith("meter_type_")]

FEATURES_ENGINEERED = (
    ["hour_sin", "hour_cos", "dow_sin", "dow_cos",   # cyclical time
     "day_of_month", "week_of_month",                # monthly position
     "is_weekend",                                    # binary day-type
     "temperature_c", "temp_sq", "heat_index"]        # weather signals
    + type_cols                                        # meter category
)

# ── 3. Train / test split  (fixed split for fair comparison) ──
X_raw  = df[FEATURES_RAW].values
X_eng  = df[FEATURES_ENGINEERED].values
y      = df[TARGET].values

X_raw_tr,  X_raw_te,  y_tr, y_te = train_test_split(X_raw,  y, test_size=0.2, random_state=42)
X_eng_tr,  X_eng_te             = train_test_split(X_eng,     test_size=0.2, random_state=42)[:2]

# Rebuild engineered split with same indices
idx_all = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.2, random_state=42)
X_raw_tr, X_raw_te = X_raw[idx_tr], X_raw[idx_te]
X_eng_tr, X_eng_te = X_eng[idx_tr], X_eng[idx_te]
y_tr, y_te         = y[idx_tr],     y[idx_te]

print(f"Train: {len(y_tr):,}  |  Test: {len(y_te):,}\n")

# ── 4. Train all four models ──────────────────────────────────

# [A] Baseline
lr_raw = LinearRegression()
lr_raw.fit(X_raw_tr, y_tr)

# [B] Linear Regression + engineered features
lr_eng = LinearRegression()
lr_eng.fit(X_eng_tr, y_tr)

# [C] Random Forest + engineered features (default settings)
rf_default = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_default.fit(X_eng_tr, y_tr)

# [D] Random Forest + engineered features (tuned)
# <- TODO  Adjust n_estimators and max_depth and observe the RMSE
rf_tuned = RandomForestRegressor(
    n_estimators = 50,      # <- TODO  try: 50, 100, 200, 300
    max_depth    = 5,       # <- TODO  try: 5, 10, 20, None (unlimited)
    min_samples_leaf = 4,    
    random_state = 42,
    n_jobs       = -1
)
rf_tuned.fit(X_eng_tr, y_tr)

# ── 5. Evaluate all models ────────────────────────────────────
def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    rmse_tr = np.sqrt(mean_squared_error(y_tr, model.predict(X_tr)))
    rmse_te = np.sqrt(mean_squared_error(y_te, model.predict(X_te)))
    r2_te   = r2_score(y_te, model.predict(X_te))
    return {"name": name, "train_rmse": rmse_tr, "test_rmse": rmse_te, "r2": r2_te}

models_results = [
    evaluate("[A] LinReg  + raw features",        lr_raw,     X_raw_tr, X_raw_te, y_tr, y_te),
    evaluate("[B] LinReg  + eng. features",        lr_eng,     X_eng_tr, X_eng_te, y_tr, y_te),
    evaluate("[C] RandFor + eng. features (def.)", rf_default, X_eng_tr, X_eng_te, y_tr, y_te),
    evaluate("[D] RandFor + eng. features (tuned)",rf_tuned,   X_eng_tr, X_eng_te, y_tr, y_te),
]

print("=" * 68)
print(f"  {'Model':<42}  {'Train RMSE':>10}  {'Test RMSE':>10}  {'R²':>6}")
print("-" * 68)
for r in models_results:
    print(f"  {r['name']:<42}  {r['train_rmse']:>10.4f}  {r['test_rmse']:>10.4f}  {r['r2']:>6.4f}")
print("=" * 68)

# ── 6. Feature importances (best model) ───────────────────────
importances = pd.Series(rf_tuned.feature_importances_, index=FEATURES_ENGINEERED)
importances = importances.sort_values(ascending=False)

# ── 7. Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Step 3 — Feature Engineering & Model Tuning Comparison",
             fontsize=13, fontweight="bold")

# — Plot A: RMSE comparison bar chart ─────────────────────────
ax = axes[0, 0]
names  = [r["name"].split("+")[0].strip() + "\n" + "+".join(r["name"].split("+")[1:]) for r in models_results]
labels = ["A\nLinReg\nraw", "B\nLinReg\neng.", "C\nRandFor\ndefault", "D\nRandFor\ntuned"]
rmses  = [r["test_rmse"] for r in models_results]
colors = ["#aec6cf", "#779ecb", "#f4a460", "#e07b39"]
bars   = ax.bar(labels, rmses, color=colors, edgecolor="black")
for bar, v in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Test RMSE (kWh) — lower is better")
ax.set_title("Model Comparison: Test RMSE")
ax.set_ylim(0, max(rmses) * 1.25)

# — Plot B: R² comparison ─────────────────────────────────────
ax = axes[0, 1]
r2s = [r["r2"] for r in models_results]
bars = ax.bar(labels, r2s, color=colors, edgecolor="black")
for bar, v in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(1.0, color="gray", linestyle=":", lw=1)
ax.set_ylabel("R² (test set) — higher is better")
ax.set_title("Model Comparison: R²")
ax.set_ylim(0, 1.08)

# — Plot C: Feature importances (tuned RF) ────────────────────
ax = axes[1, 0]
top_n = min(12, len(importances))
importances.head(top_n).plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
ax.set_xlabel("Importance")
ax.set_title(f"Top {top_n} Feature Importances\n(Tuned Random Forest)")
ax.invert_yaxis()

# — Plot D: Best model — actual vs predicted time series ──────
ax = axes[1, 1]
# Show first 7 days of a single residential meter for clarity
mask    = (df["meter_id"] == "residential_01")
df_plot = df[mask].iloc[:168].copy()          # 7 days × 24 h
X_plot  = df_plot[FEATURES_ENGINEERED].values
y_actual = df_plot[TARGET].values
y_pred_best = rf_tuned.predict(X_plot)

hours = range(168)
ax.plot(hours, y_actual,    label="Actual",           color="steelblue", lw=1.5)
ax.plot(hours, y_pred_best, label="RF Tuned Pred.",   color="orange",    lw=1.5, linestyle="--")
ax.set_xlabel("Hour (first 7 days of July — meter residential_01)")
ax.set_ylabel("energy_kwh")
ax.set_title("Best Model: Actual vs Predicted (1 week)")
ax.legend()
# Mark weekends
for day in range(7):
    if day % 7 >= 5:   # sat/sun
        ax.axvspan(day*24, (day+1)*24, alpha=0.1, color="gray")

plt.tight_layout()
plt.savefig("step3_model_comparison.png", dpi=150)
plt.show()
print("\nPlot saved -> step3_model_comparison.png\n")

# ── 8. Summary ────────────────────────────────────────────────
print("=" * 55)
print("  STEP 3 SUMMARY")
print("=" * 55)
rmse_a = models_results[0]["test_rmse"]
rmse_d = models_results[3]["test_rmse"]
print(f"  Step 1 baseline RMSE : ~{rmse_a:.3f} kWh")
print(f"  Best model RMSE      :  {rmse_d:.3f} kWh")
print(f"  Total improvement    :  {(rmse_a-rmse_d)/rmse_a*100:.1f}%")