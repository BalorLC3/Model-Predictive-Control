import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_markov_chain(driving_data, n_bins=15):
    """
    Computes the transition probability matrix for Driving Power.
    Uses Quantile-based binning (Unequally spaced).
    """
    
    # 1. Discretization (Unequally Spaced)
    # We use percentiles to define bin edges so each bin has roughly equal data initially
    # And add 0 and max to ensure coverage.
    bin_edges = np.unique(np.percentile(driving_data, np.linspace(0, 100, n_bins + 1)))
    
    # This is the value the MPC will "see"
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # 2. Assign Data to Bins
    # indices will be from 1 to n_bins (np.digitize convention)
    indices = np.digitize(driving_data, bin_edges) - 1
    
    # Clip to ensure valid indices [0, n_bins-1]
    n_real_bins = len(bin_centers)
    indices = np.clip(indices, 0, n_real_bins - 1)
    
    # 3. Build Transition Matrix
    # M[i, j] = Count of transition from bin i to bin j
    M = np.zeros((n_real_bins, n_real_bins))
    
    for k in range(len(indices) - 1):
        curr_idx = indices[k]
        next_idx = indices[k+1]
        M[curr_idx, next_idx] += 1
        
    # 4. Normalize (Row Stochastic)
    # Avoid division by zero for empty bins
    row_sums = M.sum(axis=1, keepdims=True)
    P_matrix = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums!=0)
    
    return P_matrix, bin_centers, bin_edges

# --- EXECUTION ---
if __name__ == "__main__":
    # Load your data
    try:
        data = np.load('driving_energy.npy')
    except:
        # Dummy data
        t = np.linspace(0, 1000, 10000)
        data = np.abs(np.sin(t/50)) * 20000 + np.random.normal(0, 1000, 10000)
    # Compute
    N_BINS = 15
    trans_matrix, centers, edges = compute_markov_chain(data, N_BINS)
    
    print(trans_matrix, trans_matrix.shape)
    rows, cols = trans_matrix.shape
    xpos, ypos = np.meshgrid(np.arange(cols), np.arange(rows))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    
    dx = dy = 0.8
    dz = trans_matrix.flatten()
    
    # Color logic
    colors = plt.cm.viridis(dz)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
    
    ax.set_xlabel('Next State')
    ax.set_ylabel('Current State')
    ax.set_zlabel('Prob')
    ax.set_title('3D Transition Probabilities')
    plt.show()

    # Save for MPC
    # np.savez('markov_model.npz', trans_matrix=trans_matrix, power_values=centers)
    print("Bin Centers (kW):", centers / 1000)