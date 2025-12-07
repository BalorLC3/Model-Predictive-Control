import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.cm as cm

# --- Plot Styling ---
plt.rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

def train_smart_markov(power, velocity, dt=1.0, n_clusters=15):
    """
    Computes 4 transition matrices based on Vehicle Context using K-Means.
    Contexts: 0: Braking, 1: Traffic/Idle, 2: Cruising, 3: Accelerating
    """
    # 1. K-Means Clustering on Power
    X = power.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    
    # Sort centers for physical meaning
    sorted_idx = np.argsort(centers)
    centers = centers[sorted_idx]
    map_label = {old: new for new, old in enumerate(sorted_idx)}
    mapped_labels = np.array([map_label[x] for x in labels])
    
    # 2. Calculate Acceleration
    # Insert 0 at start to match length
    accel = np.diff(velocity, prepend=velocity[0]) / dt
    
    # 3. Build 4 Matrices
    # contexts: 0=Braking, 1=Idle, 2=Cruise, 3=Accel
    matrices = np.zeros((4, n_clusters, n_clusters))
    
    for k in range(len(mapped_labels) - 1):
        curr_s = mapped_labels[k]
        next_s = mapped_labels[k+1]
        
        v = velocity[k]
        a = accel[k]
        
        # Context Classification Logic
        if a < -0.5:
            ctx = 0 # Braking
        elif v < 5.0: 
            ctx = 1 # Traffic/Idle
        elif a > 0.5:
            ctx = 3 # Accelerating
        else:
            ctx = 2 # Cruising
            
        matrices[ctx, curr_s, next_s] += 1
        
    # Normalize
    for ctx in range(4):
        row_sums = matrices[ctx].sum(axis=1, keepdims=True)
        matrices[ctx] = np.divide(matrices[ctx], row_sums, where=row_sums!=0)
        
    return matrices, centers

def plot_markov_matrices(matrices, centers):
    """
    Plots the 4 matrices in a 2x2 3D grid.
    Filters out zero-height bars to prevent RuntimeWarnings and clutter.
    """
    n_ctx, rows, cols = matrices.shape
    
    # Setup Figure with constrained_layout to fix spacing issues automatically
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    
    titles = [r"$\bf{Braking}$ ($a < -0.5 m/s^2$)", 
              r"$\bf{Idle/Traffic}$ ($v < 5 m/s$)", 
              r"$\bf{Cruising}$ (Constant Speed)", 
              r"$\bf{Accelerating}$ ($a > 0.5 m/s^2$)"]
    
    # Prepare Labels
    # Use only integer part for cleanliness if numbers are large, or 1 decimal
    tick_labels = [f"{c/1000:.1f}" for c in centers]
    tick_indices = np.arange(len(centers))

    # Grid for positions
    _x = np.arange(cols)
    _y = np.arange(rows)
    _xx, _yy = np.meshgrid(_x, _y)
    
    x_grid = _xx.flatten()
    y_grid = _yy.flatten()
    
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        matrix = matrices[i]
        dz = matrix.flatten()
        
        # --- CRITICAL FIX: Filter out zeros ---
        # Only plot bars that have a probability > 0
        mask = dz > 0.001 
        
        if np.sum(mask) > 0:
            x_plot = x_grid[mask]
            y_plot = y_grid[mask]
            z_plot = np.zeros_like(x_plot)
            dz_plot = dz[mask]
            dx = dy = 0.8
            
            # Color map
            # Normalize colors to the max height in this specific matrix for contrast
            max_height = np.max(dz_plot) if np.max(dz_plot) > 0 else 1
            colors = plt.cm.viridis(dz_plot / max_height)
            
            ax.bar3d(x_plot, y_plot, z_plot, dx, dy, dz_plot, color=colors, shade=True)
        
        # --- Formatting ---
        ax.set_title(titles[i], fontsize=14, pad=10)
        
        ax.set_xlabel('Next Power (kW)', fontsize=10, labelpad=10)
        ax.set_ylabel('Current Power (kW)', fontsize=10, labelpad=10)
        ax.set_zlabel('Probability', fontsize=10)
        ax.set_zlim(0, 1.0)
        
        # Tick Management
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels, rotation=-45, ha='left', fontsize=8)
        
        # Adjust camera angle
        ax.view_init(elev=30, azim=-60)

    # Add a main title
    fig.suptitle('Vehicle State Dependent Transition Matrices (SMPC)', fontsize=16)
    
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    try:
        p_data = np.load('driving_energy.npy')
        v_data = np.load('driving_velocity.npy')
        print("Loaded real data files.")
    except:
        print("Data files not found. Try again...")


    # 2. Compute Matrices (15 Clusters as in your logs)
    N_CLUSTERS = 16
    matrices, centers = train_smart_markov(p_data, v_data, dt=1.0, n_clusters=N_CLUSTERS)
    
    print("\nK-Means Centroids (kW):")
    print(np.round(centers/1000, 2))
    
    # 3. Plot
    plot_markov_matrices(matrices, centers)

    def find_optimal_clusters(power, max_k=30):
        """Use inertia to find elbow point"""
        inertias = []
        k_range = range(2, max_k+1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(power.reshape(-1, 1))
            inertias.append(kmeans.inertia_)
        
        # Calculate percentage variance explained
        variances = []
        for i in range(1, len(inertias)):
            variances.append((inertias[i-1] - inertias[i]) / inertias[i-1])
        
        # Find elbow (where marginal gain drops)
        for i, v in enumerate(variances):
            if i > 0 and v < 0.1:  # Less than 10% improvement
                return i + 2  # +2 because we started from k=2
        
        return 15  # Default
    
    print(find_optimal_clusters(p_data))

        