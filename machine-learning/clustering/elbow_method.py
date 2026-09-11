import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "smart_grid_july.csv"

np.random.seed(42)

# ── 1. Load & build hourly profile matrix (same as Step 1) ───
df = pd.read_csv(DATA_FILE)

hourly_profile = (
    df.groupby(["meter_id", "hour"])["energy_kwh"]
    .mean()
    .unstack(level="hour")
)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(hourly_profile)

# ── 2. Sweep k values and record inertia ──────────────────────
K_MAX = 8              # <- TODO  try 12 — does a second elbow appear?

inertias = []
k_range  = range(1, K_MAX + 1)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k = {k}   inertia = {km.inertia_:.3f}")

# ── 3. Plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 2B-2 — Elbow Method: Finding the Optimal Number of Clusters",
             fontsize=12, fontweight="bold")

# — Plot A: Elbow curve ───────────────────────────────────────
ax = axes[0]
ax.plot(list(k_range), inertias, marker="o", color="steelblue", lw=2)
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia (sum of squared distances)")
ax.set_title("Elbow Curve\n(look for the 'bend')")
ax.set_xticks(list(k_range))
ax.grid(True, alpha=0.3)

# Annotate each point with the drop from previous k
drops = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
for i, (k, drop) in enumerate(zip(list(k_range)[1:], drops)):
    ax.annotate(f"−{drop:.0f}", xy=(k, inertias[i+1]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8, color="gray")

ax.set_title("Elbow Curve\n(numbers = inertia drop from previous k)")

# — Plot B: Final clustering with elbow k, coloured profiles ──
ax = axes[1]
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
labels   = km_final.fit_predict(X_scaled)

colors   = ["steelblue", "tomato", "seagreen", "orange", "purple"]
hour_cols = list(range(24))
for c in range(3):
    idx   = np.where(labels == c)[0]
    group = hourly_profile.iloc[idx][hour_cols]
    ax.plot(hour_cols, group.mean(), lw=2.5,
            color=colors[c % len(colors)], label=f"Cluster {c}  (n={len(idx)})")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Avg energy_kwh")
ax.set_title("Hourly Profiles at k = 3\n(visual elbow pick)")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step2b2_elbow.png", dpi=150)
plt.show()
print("\nPlot saved -> step2b2_elbow.png\n")

