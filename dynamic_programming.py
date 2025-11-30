import casadi as ca
'''
================================================================================
A try-to-go file for Dynamic Programming for a generic problem, just to heat
hands.
================================================================================

'''

# ==============================================================================
# 1. PROBLEM DEFINITION (Identical)
# ==============================================================================
w = ca.SX.sym('w'); x = ca.SX.sym('x'); y = ca.SX.sym('y'); z = ca.SX.sym('z')
f = w**2 + (w - x)**2
g = x**2 + (x - y)**2
h = y**2 + (y - z)**2
total_cost = f + g + h

# ==============================================================================
# 2. METHOD 1: ANALYTICAL SOLUTION 
# ==============================================================================
def solve_analytically(z_val):
    w_opt = z_val / 13.0
    x_opt = 2 * w_opt
    y_opt = 5 * w_opt
    return w_opt, x_opt, y_opt

# ==============================================================================
# 3. METHOD 2: JOINT OPTIMIZATION 
# ==============================================================================
nlp_joint = {'f': total_cost, 'x': ca.vcat([w, x, y]), 'p': z}
solver_joint = ca.nlpsol('solver', 'ipopt', nlp_joint, {'ipopt.print_level': 0, 'print_time': 0})

def solve_jointly(z_val):
    res = solver_joint(p=z_val, x0=[0,0,0])
    w_opt, x_opt, y_opt = ca.vertsplit(res['x'])
    return float(w_opt), float(x_opt), float(y_opt)

# ==============================================================================
# 4. METHOD 3: FORWARD DYNAMIC PROGRAMMING 
# ==============================================================================
print("--- 4. Building the Forward DP Solution ---")

# --- Stage 1: min_w f(w, x) ---
w_star_f_policy = ca.Function('w_star_f', [x], [x/2])
J1_f_value = ca.substitute(f, w, w_star_f_policy(x))

# --- Stage 2: min_x [g(x, y) + J1_f(x)] ---
obj_stage2_f = g + J1_f_value
# Analytical solution: d/dx(...) = 0 => x*(y) = 2y/5
# <-- THE ONLY CHANGE IS ON THE NEXT LINE
x_star_f_policy = ca.Function('x_star_f', [y], [2*y/5])
J2_f_value = ca.substitute(obj_stage2_f, x, x_star_f_policy(y))

# --- Stage 3: min_y [h(y, z) + J2_f(y)] ---
obj_stage3_f = h + J2_f_value
nlp_stage3_f = {'f': obj_stage3_f, 'x': y, 'p': z}
solver_stage3_f = ca.nlpsol('solver', 'ipopt', nlp_stage3_f, {'ipopt.print_level': 0, 'print_time': 0})
y_star_f_policy = ca.Function('y_star_f', [z], [solver_stage3_f(p=z)['x']])

def solve_forward_dp(z_val):
    y_opt = y_star_f_policy(z_val)
    x_opt = x_star_f_policy(y_opt)
    w_opt = w_star_f_policy(x_opt)
    return float(w_opt), float(x_opt), float(y_opt)

# ==============================================================================
# 5. METHOD 4: BACKWARD DYNAMIC PROGRAMMING (Identical)
# ==============================================================================
print("--- 5. Building the Backward DP Solution ---")
y_star_b_policy = (x + z) / 3
J1_b_value = ca.substitute(g, y, y_star_b_policy) + ca.substitute(h, y, y_star_b_policy)
obj_stage2_b = f + J1_b_value
x_star_b_policy = (3*w + z) / 8
J2_b_value = ca.substitute(obj_stage2_b, x, x_star_b_policy)
obj_stage3_b = J2_b_value
nlp_stage3_b = {'f': obj_stage3_b, 'x': w, 'p': z}
solver_stage3_b = ca.nlpsol('solver', 'ipopt', nlp_stage3_b, {'ipopt.print_level': 0, 'print_time': 0})
w_star_b_policy = ca.Function('w_star_b', [z], [solver_stage3_b(p=z)['x']])

def solve_backward_dp(z_val):
    w_opt = w_star_b_policy(z_val)
    x_opt = ca.Function('x_b_func', [w, z], [x_star_b_policy])(w_opt, z_val)
    y_opt = ca.Function('y_b_func', [x, z], [y_star_b_policy])(x_opt, z_val)
    return float(w_opt), float(x_opt), float(y_opt)

# ==============================================================================
# 6. EXECUTION AND VERIFICATION (Identical)
# ==============================================================================
print("\n--- 6. Running Solvers and Comparing Results ---")
z_test_value = 14.0
print(f"Test Parameter: z = {z_test_value}\n")

w_a, x_a, y_a = solve_analytically(z_test_value)
w_j, x_j, y_j = solve_jointly(z_test_value)
w_f, x_f, y_f = solve_forward_dp(z_test_value)
w_b, x_b, y_b = solve_backward_dp(z_test_value)

print(f"{'Method':<25} | {'w':<18} | {'x':<18} | {'y':<18}")
print("-" * 75)
print(f"{'1. Analytical (Ground Truth)':<25} | {w_a:<18.6f} | {x_a:<18.6f} | {y_a:<18.6f}")
print(f"{'2. Joint Optimization':<25} | {w_j:<18.6f} | {x_j:<18.6f} | {y_j:<18.6f}")
print(f"{'3. Forward DP':<25} | {w_f:<18.6f} | {x_f:<18.6f} | {y_f:<18.6f}")
print(f"{'4. Backward DP':<25} | {w_b:<18.6f} | {x_b:<18.6f} | {y_b:<18.6f}")
print("-" * 75)

TOL = 1e-6
assert abs(w_a - w_j) < TOL, "Joint solve failed!"
assert abs(w_a - w_f) < TOL, "Forward DP failed!"
assert abs(w_a - w_b) < TOL, "Backward DP failed!"

print("\nSuccess! All methods produced the correct result.")