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
    Robust Contextual Markov Training with Global Fallback.
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


# ---------------------------
# Plotting (unchanged, uses full matrices for calculations)
# ---------------------------
def plot_markov_matrices(matrices, centers, prob_mask_threshold=0.001):
    """
    Plot the 4 context-specific matrices in a 2x2 3D grid.
    The plot masks very small probabilities for visual clarity but uses full matrices
    for any diagnostics outside plotting.
    """
    n_ctx, rows, cols = matrices.shape
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)

    titles = [
        r"$\bf{Braking}$ ($a < -$ thresh)",
        r"$\bf{Idle/Traffic}$ ($v < 5$)",
        r"$\bf{Cruising}$",
        r"$\bf{Accelerating}$ ($a >$ thresh)"
    ]

    tick_labels = [f"{c/1000:.1f}" for c in centers]
    tick_indices = np.arange(len(centers))

    _x = np.arange(cols)
    _y = np.arange(rows)
    _xx, _yy = np.meshgrid(_x, _y)
    x_grid = _xx.flatten()
    y_grid = _yy.flatten()

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        M = matrices[i].flatten()
        mask = M > prob_mask_threshold

        if mask.sum() > 0:
            x_plot = x_grid[mask]
            y_plot = y_grid[mask]
            z_plot = np.zeros_like(x_plot)
            dz_plot = M[mask]
            dx = dy = 0.8

            max_height = np.max(dz_plot) if np.max(dz_plot) > 0 else 1.0
            colors = cm.viridis(dz_plot / max_height)

            ax.bar3d(x_plot, y_plot, z_plot, dx, dy, dz_plot, color=colors, shade=True)

        ax.set_title(titles[i], fontsize=14, pad=10)
        ax.set_xlabel('Next Power (kW)', fontsize=10)
        ax.set_ylabel('Current Power (kW)', fontsize=10)
        ax.set_zlabel('Probability', fontsize=10)
        ax.set_zlim(0, 1.0)

        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels, rotation=-45, ha='left', fontsize=8)
        ax.view_init(elev=30, azim=-60)

    fig.suptitle('Vehicle State Dependent Transition Matrices (SMPC)', fontsize=16)
    plt.show()


# ---------------------------
# Diagnostics helpers
# ---------------------------
def context_trace(velocity, dt=1.0, accel_window=5, acc_thresh=0.5, brake_thresh=-0.5):
    accel = np.diff(velocity, prepend=velocity[0]) / dt
    if accel_window is not None and accel_window > 1:
        kernel = np.ones(accel_window) / accel_window
        accel = np.convolve(accel, kernel, mode='same')

    ctx = np.zeros_like(accel, dtype=int)
    for i, a in enumerate(accel):
        if a < brake_thresh:
            ctx[i] = 0
        elif velocity[i] < 5.0:
            ctx[i] = 1
        elif a > acc_thresh:
            ctx[i] = 3
        else:
            ctx[i] = 2

    return ctx, accel


def print_matrix_diagnostics(matrices):
    n_ctx = matrices.shape[0]
    for c in range(n_ctx):
        row_sums = matrices[c].sum(axis=1)
        avg_H = np.mean([entropy(matrices[c, r]) for r in range(matrices.shape[1])])
        print(f"Context {c}: row_sum_min={row_sums.min():.6f}, "
              f"row_sum_max={row_sums.max():.6f}, avg_entropy={avg_H:.4f}")


# ---------------------------
# Example main usage
# ---------------------------
if __name__ == "__main__":
    try:
        p_data = np.load("driving_energy.npy")
        v_data = np.load("driving_velocity.npy")
    except Exception as e:
        raise RuntimeError("Cannot find data files 'driving_energy.npy' and 'driving_velocity.npy'") from e

    # Train with sensible defaults; you can tweak acc/brake thresholds & smoothing
    centers, matrices = train_smart_markov(
        p_data,
        v_data,
        dt=1.0,
        n_clusters=16,
        accel_window=5,
        acc_thresh=0.5,
        brake_thresh=-0.6,
    )

    print("Cluster centers (kW):", np.round(centers / 1000.0, 2))
    plot_markov_matrices(matrices, centers)
    print_matrix_diagnostics(matrices)

    ctx, accel = context_trace(v_data, dt=1.0, accel_window=5, acc_thresh=0.5, brake_thresh=-0.6)
    unique, counts = np.unique(ctx, return_counts=True)
    print("Context counts:", dict(zip(unique, counts)))
