import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.stats import entropy
import matplotlib.cm as cm

# --- Plot Styling ---
plt.rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})


def train_smart_markov(
    power,
    velocity,
    dt=1.0,
    n_clusters=15,
    accel_window=5,
    acc_thresh=0.5,
    brake_thresh=-0.5,
    leak=0.05 
):
    """
    Scenario Markov Training.
    """
    power = np.asarray(power).reshape(-1)
    velocity = np.asarray(velocity).reshape(-1)

    # 1) KMeans clustering (1D)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=17)
    labels = kmeans.fit_predict(power.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()

    # Sort low -> high
    order = np.argsort(centers)
    centers = centers[order]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])

    # 2) Smoothed acceleration
    accel = np.diff(velocity, prepend=velocity[0]) / dt
    if accel_window is not None and accel_window > 1:
        kernel = np.ones(accel_window) / accel_window
        accel = np.convolve(accel, kernel, mode='same')

    # 3) Initialize Counts
    # Shape: [Context, From, To]
    # Context 4 will be the "GLOBAL" fallback
    counts = np.zeros((5, n_clusters, n_clusters), dtype=float)

    for t in range(len(labels) - 1):
        s0 = labels[t]
        s1 = labels[t + 1]
        v = velocity[t]
        a = accel[t]

        # Determine Context
        if a < brake_thresh:
            ctx = 0  # braking
        elif v < 5.0:
            ctx = 1  # idle
        elif a > acc_thresh:
            ctx = 3  # accelerating
        else:
            ctx = 2  # cruising

        # Add to Specific Context
        counts[ctx, s0, s1] += 1.0
        # Add to Global Context (Index 4)
        counts[4, s0, s1] += 1.0

    # 4) Normalization with Fallback Strategy
    matrices = np.zeros((4, n_clusters, n_clusters), dtype=float)
    
    # Pre-calculate Global Probabilities (Row Stochastic)
    global_probs = np.zeros((n_clusters, n_clusters))
    for r in range(n_clusters):
        row_sum = counts[4, r, :].sum()
        if row_sum > 0:
            global_probs[r, :] = counts[4, r, :] / row_sum
        else:
            global_probs[r, r] = 1.0 # True identity only if data is globally missing

    # Process each Context
    for ctx in range(4):
        for r in range(n_clusters):
            row_sum = counts[ctx, r, :].sum()
            
            if row_sum > 5.0: # If we have enough data samples for this context
                # Use the learned context probability
                probs = counts[ctx, r, :] / row_sum
            else:
                # --- FALLBACK ---
                # Not enough data for "Braking at 50kW". 
                # Use Global behavior instead of Identity.
                probs = global_probs[r, :].copy()
            
            # --- 5) DIAGONAL REDUCTION (Leakage) ---
            # If 100% probability is on diagonal, spread it slightly to neighbors
            # This helps the solver see gradients (trends)
            if leak > 0.0:
                # Add 'leak' to neighbors (r-1, r+1) and subtract from self
                # Handle boundaries
                left = max(0, r - 1)
                right = min(n_clusters - 1, r + 1)
                
                # Move probability from peak to neighbors
                # Simple logic: smooth the distribution
                current_p = probs.copy()
                probs = current_p * (1.0 - leak)
                probs[left]  += current_p[r] * (leak / 2.0)
                probs[right] += current_p[r] * (leak / 2.0)

            # Re-normalize just in case
            matrices[ctx, r, :] = probs / probs.sum()

    return centers, matrices
