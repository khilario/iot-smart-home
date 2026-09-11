import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "smart_grid_july.csv"

np.random.seed(42)

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

# ── 2. Build a feature matrix: one row per meter ──────────────
# Each meter's "fingerprint" = its average kWh for each hour (0–23)
hourly_profile = (
    df.groupby(["meter_id", "hour"])["energy_kwh"]
    .mean()
    .unstack(level="hour")   # shape: (20 meters) × (24 hours)
)

# Also keep the true meter type for validation later
meter_types = df.groupby("meter_id")["meter_type"].first()

print(f"Feature matrix shape: {hourly_profile.shape}  (meters × hours)\n")

# ── 3. Scale features ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(hourly_profile)

# ── 4. Fit K-Means ────────────────────────────────────────────
N_CLUSTERS = 3          # <- TODO  try 2, then 4 — do the clusters still make sense?

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

hourly_profile["cluster"]   = labels
hourly_profile["meter_type"] = meter_types

# ── 5. Print cluster membership ───────────────────────────────
print("=" * 48)
print("  CLUSTER ASSIGNMENTS")
print("=" * 48)
for c in range(N_CLUSTERS):
    members = hourly_profile[hourly_profile["cluster"] == c]
    types   = members["meter_type"].value_counts().to_dict()
    print(f"  Cluster {c}  ({len(members)} meters): {types}")
print()
print("  Do the clusters match the real meter types?")
print("  K-Means was never told what meter_type was.\n")

# ── 6. Plots ──────────────────────────────────────────────────
colors = ["steelblue", "tomato", "seagreen", "orange"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Step 2B-1 — K-Means Clustering  |  k = {N_CLUSTERS}",
             fontsize=13, fontweight="bold")

# — Plot A: Cluster average hourly profiles ───────────────────
ax = axes[0]
hour_cols = list(range(24))
for c in range(N_CLUSTERS):
    group = hourly_profile[hourly_profile["cluster"] == c][hour_cols]
    ax.plot(hour_cols, group.mean(), marker="o", ms=4,
            color=colors[c], lw=2, label=f"Cluster {c}")
    ax.fill_between(hour_cols,
                    group.mean() - group.std(),
                    group.mean() + group.std(),
                    alpha=0.15, color=colors[c])
ax.set_xlabel("Hour of day")
ax.set_ylabel("Avg energy_kwh")
ax.set_title("Average Hourly Profile per Cluster\n(shading = ±1 std dev)")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(True, alpha=0.3)

# — Plot B: Scatter — each meter coloured by cluster ──────────
ax = axes[1]
# Use two PCA-like axes: peak hour (14–16) vs overnight (0–5)
peak      = hourly_profile[list(range(14, 17))].mean(axis=1)
overnight = hourly_profile[list(range(0, 6))].mean(axis=1)
for c in range(N_CLUSTERS):
    mask = hourly_profile["cluster"] == c
    ax.scatter(overnight[mask], peak[mask],
               color=colors[c], s=120, edgecolors="black",
               zorder=3, label=f"Cluster {c}")
    # Annotate with meter_id
    for mid in hourly_profile[mask].index:
        ax.annotate(mid.split("_")[0][:3] + mid.split("_")[1],
                    (overnight[mid], peak[mid]),
                    fontsize=7, ha="left", va="bottom")
ax.set_xlabel("Avg overnight usage (12 am – 5 am) kWh")
ax.set_ylabel("Avg peak usage (2 pm – 4 pm) kWh")
ax.set_title("Meter Positions by Usage Pattern\n(clusters coloured)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step2b1_clusters.png", dpi=150)
plt.show()
print("Plot saved -> step2b1_clusters.png\n")

