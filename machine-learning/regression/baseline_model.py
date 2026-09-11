import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "smart_grid_july.csv"

# ── 0. Reproducibility ───────────────────────────────────────
np.random.seed(42)

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
print(f"Full dataset: {len(df):,} rows\n")

# ── 2. Select a small training slice ─────────────────────────
N_TRAIN_ROWS = 1000          # <- TODO  try 500, 2 000 and note the RMSE

df_subset = df.iloc[:N_TRAIN_ROWS].copy()
print(f"Using first {N_TRAIN_ROWS:,} rows for training+testing.")

# ── 3. Feature selection (deliberately minimal) ───────────────
FEATURES = ["hour", "day_of_week", "temperature_c"]   # only 3 features
TARGET   = "energy_kwh"

X = df_subset[FEATURES]
y = df_subset[TARGET]

# ── 4. Train / test split (80 / 20) ──────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}\n")

# ── 5. Train the model ────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=" * 42)
print("  MODEL PERFORMANCE  (test set)")
print("=" * 42)
print(f"  RMSE : {rmse:.4f} kWh")
print(f"  MAE  : {mae:.4f} kWh")
print(f"  R²   : {r2:.4f}")
print("=" * 42)
print()
print("  RMSE = Root Mean Squared Error  (lower is better)")
print("  MAE  = Mean Absolute Error       (lower is better)")
print("  R²   = Coefficient of Determination  (1.0 is perfect)\n")

# ── 7. Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Step 1 — Baseline Linear Regression  |  Train rows: {N_TRAIN_ROWS:,}  |  "
    f"RMSE: {rmse:.3f} kWh  |  R²: {r2:.3f}",
    fontsize=12, fontweight="bold"
)

# — Plot A: Actual vs Predicted scatter ───────────────────────
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.4, s=18, color="steelblue", edgecolors="none")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual energy_kwh")
ax.set_ylabel("Predicted energy_kwh")
ax.set_title("Actual vs Predicted")
ax.legend()

# — Plot B: Residuals ─────────────────────────────────────────
residuals = y_test.values - y_pred
ax = axes[1]
ax.scatter(y_pred, residuals, alpha=0.4, s=18, color="coral", edgecolors="none")
ax.axhline(0, color="black", lw=1.2, linestyle="--")
ax.set_xlabel("Predicted energy_kwh")
ax.set_ylabel("Residual (actual − predicted)")
ax.set_title("Residual Plot\n(spread = unexplained variance)")

# — Plot C: Feature coefficients ──────────────────────────────
ax = axes[2]
coefs = pd.Series(model.coef_, index=FEATURES)
colors = ["green" if c > 0 else "tomato" for c in coefs]
coefs.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
ax.set_title("Linear Regression Coefficients\n(feature influence)")
ax.set_ylabel("Coefficient value")
ax.set_xticklabels(FEATURES, rotation=15)
ax.axhline(0, color="black", lw=0.8)

plt.tight_layout()
plt.savefig("step1_baseline.png", dpi=150)
plt.show()
print("Plot saved -> step1_baseline.png\n")

# ── 8. Quick visual: actual vs predicted over time (first 72 h) ──
fig2, ax2 = plt.subplots(figsize=(14, 4))
sample = df_subset.iloc[:72].copy()
X_s = sample[FEATURES]
y_s = sample[TARGET]
y_pred_s = model.predict(X_s)

ax2.plot(range(72), y_s.values,  label="Actual",    color="steelblue", lw=1.5)
ax2.plot(range(72), y_pred_s,    label="Predicted", color="orange",    lw=1.5, linestyle="--")
ax2.set_xlabel("Hour (first 3 days of July)")
ax2.set_ylabel("energy_kwh")
ax2.set_title("Step 1 — Actual vs Predicted: first 72 hours")
ax2.legend()
plt.tight_layout()
plt.savefig("step1_timeseries.png", dpi=150)
plt.show()
print("Plot saved -> step1_timeseries.png\n")


