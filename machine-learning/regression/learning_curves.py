import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "smart_grid_july.csv"

np.random.seed(42)

# ── 1. Load full dataset ──────────────────────────────────────
df = pd.read_csv(DATA_FILE)
print(f"Full dataset loaded: {len(df):,} rows\n")

# ── 2. Features (same as Step 1 — intentionally unchanged) ───
FEATURES = ["hour", "day_of_week", "temperature_c"]
TARGET   = "energy_kwh"

X_full = df[FEATURES].values
y_full = df[TARGET].values

# Shuffle BEFORE splitting so the test set contains all meter types
# (the CSV is ordered by meter_id, so the last rows are all industrial)
shuffle_idx = np.random.permutation(len(y_full))
X_full = X_full[shuffle_idx]
y_full = y_full[shuffle_idx]

# Hold out a FIXED test set — comparisons across training sizes stay fair
TEST_ROWS  = 2_000
X_test     = X_full[-TEST_ROWS:]
y_test     = y_full[-TEST_ROWS:]
X_pool     = X_full[:-TEST_ROWS]
y_pool     = y_full[:-TEST_ROWS]

print(f"Fixed test set : {TEST_ROWS:,} rows  (never used for training)")
print(f"Training pool  : {len(X_pool):,} rows\n")

# ── 3. Training sizes to sweep ────────────────────────────────
TRAIN_SIZES = [ 100, 250, 500, 750, 1_000, 1_500, 2_000,
               3_000, 4_000, 5_000, 6_000, 8_000, 10_000, 12_000]

results = []

print(f"{'Train rows':>12}  {'Train RMSE':>11}  {'Test RMSE':>10}  {'Test R²':>8}")
print("-" * 48)

for n in TRAIN_SIZES:
    if n > len(X_pool):
        print(f"  {n:>10,}  — skipped (not enough rows in pool)")
        continue

    X_tr = X_pool[:n]
    y_tr = y_pool[:n]

    model = LinearRegression()
    model.fit(X_tr, y_tr)

    train_rmse = np.sqrt(mean_squared_error(y_tr, model.predict(X_tr)))
    test_rmse  = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    test_r2    = r2_score(y_test, model.predict(X_test))

    results.append({"n": n, "train_rmse": train_rmse, "test_rmse": test_rmse, "r2": test_r2})
    print(f"  {n:>10,}  {train_rmse:>11.4f}  {test_rmse:>10.4f}  {test_r2:>8.4f}")

results_df = pd.DataFrame(results)

# ── 4. Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step 2 — Learning Curve: How Training Size Affects Error",
             fontsize=13, fontweight="bold")

# — Plot A: Learning curve (RMSE vs training size) ────────────
ax = axes[0]
ax.plot(results_df["n"], results_df["train_rmse"],
        marker="o", color="steelblue", label="Train RMSE", lw=2)
ax.plot(results_df["n"], results_df["test_rmse"],
        marker="s", color="tomato",    label="Test RMSE",  lw=2)
ax.set_xlabel("Number of training rows")
ax.set_ylabel("RMSE (kWh)")
ax.set_title("Learning Curve\n(where does it stop improving?)")
ax.legend()
ax.grid(True, alpha=0.3)

# Detect if curve is flat from the very start (feature ceiling, not data ceiling)
rmse_range = results_df["test_rmse"].max() - results_df["test_rmse"].min()
if rmse_range < 1.0:
    
    ax.text(0.2, 0.80,
            "Curve is flat from the start\n Why? (hint: it has to do with features\n",
            transform=ax.transAxes, fontsize=9, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="darkred"))
    
else:
    plateau_start = results_df.loc[results_df["n"] >= 4_000, "n"].min()
    if pd.notna(plateau_start):
        ax.axvline(plateau_start, color="green", linestyle="--", lw=1.5)
        ax.text(plateau_start + 100, results_df["test_rmse"].max() * 0.98,
                "← plateau region", color="green", fontsize=9)

# — Plot B: R² vs training size ───────────────────────────────
ax = axes[1]
ax.plot(results_df["n"], results_df["r2"],
        marker="D", color="purple", lw=2)
ax.axhline(1.0, color="gray", linestyle=":", lw=1, label="Perfect R²=1")
ax.set_xlabel("Number of training rows")
ax.set_ylabel("R² score (test set)")
ax.set_title("Model Quality vs Training Size\n(1.0 = perfect fit)")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step2_learning_curve.png", dpi=150)
plt.show()
print("\nPlot saved -> step2_learning_curve.png\n")