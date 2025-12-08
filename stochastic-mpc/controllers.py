import casadi as ca
import numpy as np
from abc import ABC, abstractmethod
from casadi_ode_solver import rk4_step_ca
from sys_dynamics_casadi import SystemParameters
from collections import deque
from markov_chain import train_smart_markov

class BaseController(ABC):
    @abstractmethod
    def compute_control(self, state, disturbance, velocity=0.0):
        pass

class Thermostat(BaseController):
    '''Rule based controller; threshold based'''
    def __init__(self):
        self.cooling_active = False

    def compute_control(self, state, disturbance, velocity=0.0):
        T_batt, _, _ = state
        T_upper, T_lower = 34.0, 32.5
        
        # Optimización: Lógica nativa de Python para simulación numérica
        if isinstance(T_batt, (float, np.floating, int)):
            if T_batt > T_upper:
                self.cooling_active = True
            elif T_batt < T_lower:
                self.cooling_active = False
            w_pump = 2000.0 if self.cooling_active else 0.0
            w_comp = 3000.0 if self.cooling_active else 0.0
        else:
            # Lógica CasADi para generación del grafo
            self.cooling_active = ca.if_else(
                T_batt > T_upper,
                True,
                ca.if_else(T_batt < T_lower, False, self.cooling_active)
            )
            w_pump = ca.if_else(self.cooling_active, 2000.0, 0.0)
            w_comp = ca.if_else(self.cooling_active, 3000.0, 0.0)
        return w_comp, w_pump

    # minimize
    # J = \mathbb{E}_{P_driv} \left[\sum_{k=0}^{N_p-1} P_{comp}(k) + P_{pump}(k) + \alpha(T_{batt}(N_p) - T_{batt,des})^2 \right]
    # subject to
    # T_{batt,min} \le T_{batt}(k) \le T_{batt,max}
    # T_{clnt,min} \le T_{clnt}(k) \le T_{clnt,max}
    # w_{comp,min} \le w_{comp}(k) \le w_{comp,max}
    # w_{pump,min} \le w_{pump}(k) \le w_{pump,max}
    # P_{batt,min} \le P_{batt}(k) \le P_{batt,max}

class DMPC(BaseController):
    def __init__(self, dt=1.0, T_des=32.5, horizon=20, alpha=1.0, avg_window=15):
        """
        Deterministic MPC with Recursive Moving Average Filter.
        Predicts future power is constant = average of last 'n' seconds.
        """
        self.dt = dt
        self.N = horizon
        self.params = SystemParameters()
        
        # --- Filter Config ---
        self.n_window = int(avg_window)
        self.p_driv_history = deque(maxlen=self.n_window)
        self.prev_avg = 0.0
        
        self.current_step_idx = 0
        
        # Dimensions
        self.n_x = 3 * (self.N + 1)
        self.n_u = 2 * self.N
        self.n_slack = 2 * (self.N + 1)

        # Constraints
        self.T_mins = np.array([30.0, 28.0]) 
        self.T_maxs = np.array([35.0, 34.0])
        self.w_mins = np.array([0.0, 0.0])   
        self.w_maxs = np.array([10000.0, 10000.0])
        self.P_batt_min = -200.0
        self.P_batt_max = 200.0 

        # Parameters
        self.alpha = alpha
        self.T_des = T_des
        self.rho_soft = 0.2 # Low penalty for fairness

        # Build Solver
        print("Compiling DMPC Solver...")
        self._build_solver()
        print("Done.")

        self.x_guess = np.zeros(self.n_x)
        self.u_guess = np.zeros(self.n_u)
        self.slack_guess = np.zeros(self.n_slack)

    def _build_solver(self):
        X = ca.MX.sym('X', 3, self.N + 1) 
        U = ca.MX.sym('U', 2, self.N)     
        S = ca.MX.sym('S', 2, self.N + 1) 
        P_x0 = ca.MX.sym('P_x0', 3)       
        P_dist = ca.MX.sym('P_dist', 2, self.N) 
        
        obj = 0
        g = []
        lbg = []
        ubg = []
        zeros_3 = np.zeros(3)

        g.append(X[:, 0] - P_x0)
        lbg.append(zeros_3); ubg.append(zeros_3)

        for k in range(self.N):
            x_next, diag = rk4_step_ca(X[:, k], U[:, k], P_dist[:, k], self.params, self.dt)
            g.append(x_next - X[:, k+1])
            lbg.append(zeros_3); ubg.append(zeros_3)
            
            P_batt_kW = diag[1] / 1000.0
            P_comp_kW = diag[7] / 1000.0
            
            # Energy Cost
            obj += (P_batt_kW + P_comp_kW) * (self.dt / 3600.0)
            # Slack Cost
            obj += self.rho_soft * (S[0, k]**2 + S[1, k]**2)
            
            # Hard Power
            g.append(P_batt_kW); lbg.append([self.P_batt_min]); ubg.append([self.P_batt_max])
            
            # Soft Temp (Upper)
            g.append(X[0, k] - S[0, k]); ubg.append([self.T_maxs[0]]); lbg.append([-ca.inf])
            # Soft Temp (Lower)
            g.append(X[0, k] + S[1, k]); lbg.append([self.T_mins[0]]); ubg.append([ca.inf])

        obj += self.alpha * (X[0, self.N] - self.T_des)**2
        
        # Bounds (Loose)
        lbx_X = np.tile([-ca.inf, -ca.inf, -ca.inf], self.N + 1)
        ubx_X = np.tile([ca.inf, ca.inf, ca.inf], self.N + 1)
        lbx_U = np.tile(self.w_mins, self.N)
        ubx_U = np.tile(self.w_maxs, self.N)
        lbx_S = np.zeros(self.n_slack); ubx_S = np.full(self.n_slack, ca.inf)
        
        self.lbx = np.concatenate([lbx_X, lbx_U, lbx_S])
        self.ubx = np.concatenate([ubx_X, ubx_U, ubx_S])
        self.lbg = np.concatenate(lbg); self.ubg = np.concatenate(ubg)

        OPT_vars = ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(S))
        OPT_params = ca.vertcat(P_x0, ca.vec(P_dist))
        nlp = {'f': obj, 'x': OPT_vars, 'g': ca.vertcat(*g), 'p': OPT_params}
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.tol': 1e-4, 
            'expand': True,
        }
        self.solver = ca.nlpsol('NMPC', 'ipopt', nlp, opts)

    def compute_control(self, state, current_disturbance, velocity=0.0):
        current_P_driv = current_disturbance[0]
        current_T_amb = current_disturbance[1]
        
        # Filter Logic
        if len(self.p_driv_history) == self.n_window:
            val_k = current_P_driv
            val_k_minus_n = self.p_driv_history[0] 
            current_avg = self.prev_avg + (val_k - val_k_minus_n) / self.n_window
            self.p_driv_history.append(val_k)
        else:
            self.p_driv_history.append(current_P_driv)
            current_avg = np.mean(self.p_driv_history)
        
        self.prev_avg = current_avg

        # Prediction: Constant Average
        p_driv_horizon = np.full(self.N, current_avg)
        t_amb_horizon = np.full(self.N, current_T_amb)
        
        d_horizon = np.vstack([p_driv_horizon, t_amb_horizon])
        d_flat = d_horizon.flatten(order='F')
        
        p_val = np.concatenate([state, d_flat])
        x0_val = np.concatenate([self.x_guess, self.u_guess, self.slack_guess])
        
        try:
            res = self.solver(x0=x0_val, p=p_val, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
            opt_var = res['x'].full().flatten()
            u_opt = opt_var[self.n_x : self.n_x + 2]
            
            # Warm Start
            idx_u = self.n_x; idx_s = self.n_x + self.n_u
            x_traj = opt_var[:idx_u].reshape(3, self.N+1)
            u_traj = opt_var[idx_u:idx_s].reshape(2, self.N)
            s_traj = opt_var[idx_s:].reshape(2, self.N+1)
            
            self.x_guess = np.hstack([x_traj[:, 1:], x_traj[:, -1:]]).flatten()
            self.u_guess = np.hstack([u_traj[:, 1:], u_traj[:, -1:]]).flatten()
            self.slack_guess = np.hstack([s_traj[:, 1:], s_traj[:, -1:]]).flatten()
            
            self.current_step_idx += 1
            return u_opt
            
        except Exception as e:
            # Fallback
            self.x_guess = np.zeros(self.n_x)
            return np.array([5000.0, 5000.0])


class SMPC(BaseController):
    def __init__(self, driving_power, driving_velocity, dt=1.0, T_des=32.5, horizon=20, alpha=1.0, n_clusters=15):
        """
        SMPC (Expected Value).
        Uses Contextual Markov Chain (Velocity/Accel aware) to predict Expected Power. Note: Normally SMPC is classified in three, we
        use Expected Value SMPC, which is not Chance-Constrained SMPC nor
        Multiple-Scenario SMPC, which are heavier to compute
        """
        self.dt = dt
        self.N = horizon
        self.params = SystemParameters()
        self.n_clusters = n_clusters
        
        print(f"Training Smart Markov ({n_clusters} clusters)...")
        self.centers, self.matrices = train_smart_markov(driving_power, driving_velocity, dt, n_clusters)
        print("Done.")
        
        self.current_step_idx = 0
        self.prev_velocity = 0.0 
        
        # Dimensions
        self.n_x = 3 * (self.N + 1)
        self.n_u = 2 * self.N
        self.n_slack = 2 * (self.N + 1)

        # Constraints
        self.T_mins = np.array([30.0, 28.0]) 
        self.T_maxs = np.array([35.0, 34.0])
        self.w_mins = np.array([0.0, 0.0])   
        self.w_maxs = np.array([10000.0, 10000.0])
        self.P_batt_min = -200.0
        self.P_batt_max = 200.0 
        
        self.alpha = alpha
        self.T_des = T_des
        
        # Equal to DMPC for fair comparison
        self.rho_soft = 0.2 

        print("Compiling SMPC Solver...")
        self._build_solver()
        print("Done.")

        self.x_guess = np.zeros(self.n_x)
        self.u_guess = np.zeros(self.n_u)
        self.slack_guess = np.zeros(self.n_slack)

    def _build_solver(self):
        X = ca.MX.sym('X', 3, self.N + 1) 
        U = ca.MX.sym('U', 2, self.N)     
        S = ca.MX.sym('S', 2, self.N + 1) 
        P_x0 = ca.MX.sym('P_x0', 3)       
        # Input: Expected Disturbance Trajectory
        P_dist_expected = ca.MX.sym('P_dist', 2, self.N) 
        
        obj = 0
        g = []
        lbg = []
        ubg = []
        zeros_3 = np.zeros(3)

        g.append(X[:, 0] - P_x0)
        lbg.append(zeros_3); ubg.append(zeros_3)

        for k in range(self.N):
            x_next, diag = rk4_step_ca(X[:, k], U[:, k], P_dist_expected[:, k], self.params, self.dt)
            g.append(x_next - X[:, k+1])
            lbg.append(zeros_3); ubg.append(zeros_3)
            
            P_batt_kW = diag[1] / 1000.0
            P_comp_kW = diag[7] / 1000.0
            
            # Expected Cost
            obj += (P_batt_kW + P_comp_kW) * (self.dt / 3600.0)
            obj += self.rho_soft * (S[0, k]**2 + S[1, k]**2)
            
            # Constraints (Applied on Expected Trajectory)
            g.append(P_batt_kW); lbg.append([self.P_batt_min]); ubg.append([self.P_batt_max])
            
            g.append(X[0, k] - S[0, k]); ubg.append([self.T_maxs[0]]); lbg.append([-np.inf])
            g.append(X[0, k] + S[1, k]); lbg.append([self.T_mins[0]]); ubg.append([np.inf])

        obj += self.alpha * (X[0, self.N] - self.T_des)**2
        
        # --- SAFETY BOX BOUNDS ---
        # Temp: -10 to 80 (Physics break down outside this)
        x_min_safe = [-10.0, -10.0, -10.0]
        x_max_safe = [80.0, 80.0, 10.0]
        lbx_X = np.tile(x_min_safe, self.N + 1)
        ubx_X = np.tile(x_max_safe, self.N + 1)
        
        lbx_U = np.tile(self.w_mins, self.N)
        ubx_U = np.tile(self.w_maxs, self.N)
        
        # Bound the Slack to reasonable values (e.g., max 5 deg violation)
        # If Inf, solver exploits low rho to generate NaNs.
        lbx_S = np.zeros(self.n_slack)
        ubx_S = np.full(self.n_slack, 5.0) 
        
        self.lbx = np.concatenate([lbx_X, lbx_U, lbx_S])
        self.ubx = np.concatenate([ubx_X, ubx_U, ubx_S])
        self.lbg = np.concatenate(lbg); self.ubg = np.concatenate(ubg)

        OPT_vars = ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(S))
        OPT_params = ca.vertcat(P_x0, ca.vec(P_dist_expected))
        
        nlp = {'f': obj, 'x': OPT_vars, 'g': ca.vertcat(*g), 'p': OPT_params}
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.tol': 1e-4, 
            'expand': True,
            'ipopt.bound_mult_init_method': 'mu-based' # Stability fix
        }
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4, 'expand': True}
        self.solver = ca.nlpsol('ExpectedSMPC', 'ipopt', nlp, opts)

    def compute_control(self, state, current_disturbance, velocity=0.0):
        curr_P = current_disturbance[0]
        curr_T_amb = current_disturbance[1]
        
        # 1. Determine Context (Smart Markov)
        if self.current_step_idx == 0:
            accel = 0.0
        else:
            accel = (velocity - self.prev_velocity) / self.dt
        self.prev_velocity = velocity
        
        # Select Matrix
        if accel < -0.5: ctx = 0 # Brake
        elif velocity < 5.0: ctx = 1 # Idle
        elif accel > 0.5: ctx = 3 # Accel
        else: ctx = 2 # Cruise
            
        P_matrix = self.matrices[ctx]
        
        # 2. Identify Current Cluster
        cluster_idx = (np.abs(self.centers - curr_P)).argmin()
        
        # 3. Propagate Expected Value
        pi_vec = np.zeros(self.n_clusters); pi_vec[cluster_idx] = 1.0
        expected_power = np.zeros(self.N)
        
        for k in range(self.N):
            expected_power[k] = np.dot(pi_vec, self.centers)
            pi_vec = pi_vec @ P_matrix
            
        t_amb_horizon = np.full(self.N, curr_T_amb)
        
        # 4. Solve
        p_inputs_horizon = np.vstack([expected_power, t_amb_horizon])
        p_flat = p_inputs_horizon.flatten(order='F')
        
        p_val = np.concatenate([state, p_flat])
        x0_val = np.concatenate([self.x_guess, self.u_guess, self.slack_guess])
        
        try:
            res = self.solver(x0=x0_val, p=p_val, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
            
            # Robust extraction
            if not self.solver.stats()['success']:
                # Optimization failed partially, but result might be usable.
                # Reset warm start to prevent error propagation.
                self.x_guess = np.zeros(self.n_x)
            else:
                opt_var = res['x'].full().flatten()
                idx_u = self.n_x; idx_s = self.n_x + self.n_u
                
                # Update Warm Start
                x_traj = opt_var[:idx_u].reshape(3, self.N+1)
                u_traj = opt_var[idx_u:idx_s].reshape(2, self.N)
                s_traj = opt_var[idx_s:].reshape(2, self.N+1)
                
                self.x_guess = np.hstack([x_traj[:,1:], x_traj[:,-1:]]).flatten()
                self.u_guess = np.hstack([u_traj[:,1:], u_traj[:,-1:]]).flatten()
                self.slack_guess = np.hstack([s_traj[:,1:], s_traj[:,-1:]]).flatten()

            return res['x'].full().flatten()[self.n_x : self.n_x + 2]
            
        except Exception:
            self.x_guess = np.zeros(self.n_x)
            self.current_step_idx += 1
            return np.array([5000.0, 5000.0])