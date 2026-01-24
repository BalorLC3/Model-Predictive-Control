import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from MPC import MPC_casadi      
from lq_gain import lq_gain            # returns K, P
from computations import compute_VT    # terminal set, optional

# ==========================================================
# System definition (Example 2.1 / 2.3 – Cannon, 2016)
# ==========================================================
A = np.array([[1.1, 2.0],
              [0.0, 0.95]])
B = np.array([[0.0],
              [0.0787]])
C = np.array([[-1.0, 1.0]])
Q = C.T @ C
R = np.array([[1.0]])

# Constraints
F = np.array([[0, 1/8],
              [1/8, 0],
              [0, -1/8],
              [-1/8, 0],
              [0, 0],
              [0, 0]])
G = np.array([0, 0, 0, 0, 1, -1]).reshape(-1, 1)

# Parameters
x0 = np.array([-7.5, 0.5])   
N = 6
total_steps = 30

# LQR terminal gain (Mode 2)
P = solve_discrete_are(A, B, Q, R)
K, _ = lq_gain(A, B, Q, R)
VT = compute_VT(F, G, K, A, B)

# Closed-loop MPC simulation (re-optimizing every step)
x_current = x0.copy()
x_history = [x_current]
u_history = []
predicted_trajectories = []
predicted_controls = []

for step in range(total_steps):
    cost, u_opt, c_opt, x_opt = MPC_casadi(F, G, A, B, Q, R, VT, x_current, N=N)
    if step == 0:
        cost_x0 = cost
    if u_opt is None:
        print(f"MPC failed at step {step}")
        break
    print(f"MPC step {step}: u*={u_opt:.3f}")
    u_current = u_opt
    u_history.append(u_current)

    # save prediction (for plotting)
    if x_opt is not None and c_opt is not None:
        predicted_trajectories.append({'step': step, 'x_pred': x_opt.copy()})
        u_predicted = [(K @ x_opt[:, k] + c_opt[k]).item() for k in range(len(c_opt))]
        predicted_controls.append({'step': step, 'u_pred': u_predicted})

    x_next = A @ x_current.reshape(-1, 1) + B * u_current
    x_current = x_next.flatten()
    x_history.append(x_current)
x_closed_loop = np.array(x_history).T
u_closed_loop = np.array(u_history)

# Open-loop MPC prediction (Mode 1 until N, then Mode 2 LQR)
x_open_loop = [x0.copy()]
u_open_loop = []

x_pred0 = predicted_trajectories[0]['x_pred']
u_pred0 = predicted_controls[0]['u_pred']

x_temp = x0.copy()

def cost_closed_lqr(x, u, Q, R):
    '''Closed-loop cost Jcl'''
    cost_cl = 0
    for i in range(len(u)):
        cost_cl += x[:, i].T @ Q @ x[:, i] + u[i] * R * u[i]
    return cost_cl.item()

# --- Mode 1: MPC-predicted inputs (k = 0,...,N-1)
for k in range(N):
    u_ol = u_pred0[k]
    u_open_loop.append(u_ol)
    x_temp = A @ x_temp.reshape(-1, 1) + B * u_ol
    x_open_loop.append(x_temp.flatten())

# --- Mode 2: LQR continuation (k = N,...,total_steps-1)
for k in range(N, total_steps):
    u_ol = -K @ x_temp
    u_open_loop.append(u_ol.item())
    x_temp = A @ x_temp.reshape(-1, 1) + B * u_ol
    x_open_loop.append(x_temp.flatten())

x_open_loop_arr = np.array(x_open_loop).T
u_open_loop_arr = np.array(u_open_loop)

cost_cl_x0 = cost_closed_lqr(x_open_loop_arr, u_open_loop_arr,
                      Q, R)

print("J*(x0):   ", cost_x0)
print("Jcl*(x0): ", cost_cl_x0)


# Plot 1: State trajectories (Fig. 2.4 style)
plt.figure(figsize=(9,6))
plt.plot(x_closed_loop[0, :], x_closed_loop[1, :], 'o-', color='black',
         label='Closed-loop (re-optimizing)', markersize=4, linewidth=2)
plt.plot(x_open_loop_arr[0, :], x_open_loop_arr[1, :], 's--', color='red',
         label='Open-loop (initial MPC + LQR at N)', markersize=4, alpha=0.8, linewidth=2)
if predicted_trajectories:
    x_pred = predicted_trajectories[0]['x_pred']
    plt.plot(x_pred[0, :], x_pred[1, :], ':', color='blue',
             label='Initial MPC prediction (Mode 1)', linewidth=2, alpha=0.6)

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Example 2.3 - Open-loop vs Closed-loop MPC')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 2: Control inputs (Fig. 2.5 style)
plt.figure(figsize=(10,5))
plt.step(range(len(u_closed_loop)), u_closed_loop, where='post',
         label='Closed-loop (re-optimizing)', color='black', linewidth=2)
plt.step(range(len(u_open_loop_arr)), u_open_loop_arr, where='post',
         label='Open-loop (initial MPC + LQR at N)', color='red', linestyle='--', linewidth=2, alpha=0.8)
plt.axvline(x=N-1, color='gray', linestyle='--', linewidth=1, label='Switch to Mode 2 (LQR)')
plt.axhline(1, color='gray', linestyle=':', linewidth=1, alpha=0.6)
plt.axhline(-1, color='gray', linestyle=':', linewidth=1, alpha=0.6)
plt.xlabel('Time step  $k$')
plt.ylabel('Control input  $u(k)$')
plt.title('Example 2.3 – Control Inputs')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
