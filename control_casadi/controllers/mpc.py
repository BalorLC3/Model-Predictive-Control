import casadi as ca
import numpy as np
from system.casadi_ode_solver import rk4_step_ca
from system.sys_dynamics_casadi import SystemParameters
from collections import deque
from controllers.markov_chain import train_smart_markov
from controllers.base import BaseController
import time
    # minimize
    # J = \mathbb{E}_{P_driv} \left[\sum_{k=0}^{N_p-1} P_{comp}(k) + P_{pump}(k) + \alpha(T_{batt}(N_p) - T_{batt,des})^2 \right]
    # subject to
    # T_{batt,min} \le T_{batt}(k) \le T_{batt,max}
    # T_{clnt,min} \le T_{clnt}(k) \le T_{clnt,max}
    # w_{comp,min} \le w_{comp}(k) \le w_{comp,max}
    # w_{pump,min} \le w_{pump}(k) \le w_{pump,max}
    # P_{batt,min} \le P_{batt}(k) \le P_{batt,max}


class DMPC(BaseController):
    def __init__(self, dt=1.0, T_des=33.0, horizon=20, alpha=1.0, avg_window=15):
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

        # Constraints
        self.T_mins = np.array([15.0, 15.0]) 
        self.T_maxs = np.array([45.0, 45.0])
        self.w_mins = np.array([0.0, 0.0])   
        self.w_maxs = np.array([10000.0, 10000.0])

        self.P_batt_min = -200
        self.P_batt_max = 200

        # Parameters
        self.alpha = alpha
        self.T_des = T_des

        # Build Solver
        print("Compiling DMPC Solver...")
        self._build_solver()
        print("Done.")

        self.x_guess = np.zeros(self.n_x)
        self.u_guess = np.zeros(self.n_u)

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
            P_cool_kW = diag[0] / 1000 / 3600
            P_batt_W = diag[1] / 1000
            
            
            # Energy Cost
            obj += 3.0 * (P_cool_kW * self.dt)

            # Hard Power
            g.append(P_batt_W); lbg.append([self.P_batt_min]); ubg.append([self.P_batt_max])
            # Hard Temp
            g.append(X[0, k]); lbg.append([self.T_mins[0]]); ubg.append([self.T_maxs[0]])

        obj += self.alpha * (X[0, self.N] - self.T_des)**2
        
        # Bounds (Loose)
        lbx_X = np.tile([-ca.inf, -ca.inf, -ca.inf], self.N + 1)
        ubx_X = np.tile([ca.inf, ca.inf, ca.inf], self.N + 1)
        lbx_U = np.tile(self.w_mins, self.N)
        ubx_U = np.tile(self.w_maxs, self.N)
        
        self.lbx = np.concatenate([lbx_X, lbx_U])
        self.ubx = np.concatenate([ubx_X, ubx_U])
        self.lbg = np.concatenate(lbg); self.ubg = np.concatenate(ubg)

        OPT_vars = ca.vertcat(ca.vec(X), ca.vec(U))
        OPT_params = ca.vertcat(P_x0, ca.vec(P_dist))
        nlp = {'f': obj, 'x': OPT_vars, 'g': ca.vertcat(*g), 'p': OPT_params}
        opts = {
            'ipopt.print_level': 0, # Change from 0 or 1 to 5 for maximum detail
            'print_time': 0, 
            'ipopt.tol': 1e-4, 
            'expand': True,
            'ipopt.bound_mult_init_method': 'mu-based'
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
        x0_val = np.concatenate([self.x_guess, self.u_guess])
        
        res = self.solver(x0=x0_val, p=p_val, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)            # Robust extraction
        stats = self.solver.stats()

        x_opt = None
        if 'x' in res:
            x_opt = res['x'].full().flatten()

        if stats['success'] or x_opt is not None:
            if not stats['success']:
                print(f"Warning: {type(self).__name__} using suboptimal solution at step {self.current_step_idx}")
                print("return_status:", stats.get('return_status', 'unknown'))

            n_x_vars = self.n_x
            n_u_vars = self.n_u

            x_traj = x_opt[0:n_x_vars].reshape((3, self.N + 1), order='F')
            u_traj = x_opt[n_x_vars:n_x_vars + n_u_vars].reshape((2, self.N), order='F')

            # Shift warm start
            self.x_guess = np.hstack([x_traj[:, 1:], x_traj[:, -1:]]).flatten(order='F')
            self.u_guess = np.hstack([u_traj[:, 1:], u_traj[:, -1:]]).flatten(order='F')

            u_opt = u_traj[:, 0]

        else:
            print(f"ERROR: {type(self).__name__} no primal solution at step {self.current_step_idx}")
            u_opt = self.u_guess[:2] if self.u_guess.size >= 2 else np.array([5000.0, 5000.0])


        self.current_step_idx += 1 
        return u_opt


class SMPC(BaseController):
    def __init__(self, driving_power, driving_velocity, dt=1.0, T_des=33.0, horizon=20, alpha=1.0, n_clusters=15):
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
        start = time.time()
        self.centers, self.matrices = train_smart_markov(driving_power, driving_velocity, dt, n_clusters)
        train_time = time.time() - start 
        print(f"Total training time: {train_time:.3f} s")
        
        self.current_step_idx = 0
        self.prev_velocity = 0.0 
        
        # Dimensions
        self.n_x = 3 * (self.N + 1)
        self.n_u = 2 * self.N

        # Constraints
        self.T_mins = np.array([15.0, 15.0]) 
        self.T_maxs = np.array([45.0, 45.0])
        self.w_mins = np.array([0.0, 0.0])   
        self.w_maxs = np.array([10000.0, 10000.0])

        self.P_batt_min = -200
        self.P_batt_max = 200

        self.alpha = alpha
        self.T_des = T_des
        

        print("Compiling SMPC Solver...")
        self._build_solver()
        print("Done.")

        self.x_guess = np.zeros(self.n_x)
        self.u_guess = np.zeros(self.n_u)

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

            P_cool_kW = diag[0] / 1000 / 3600
            P_batt_kW = diag[1] / 1000 
            # Expected Cost
            obj += 2.0 * (P_cool_kW * self.dt)
            
            # Constraints (Applied on Expected Trajectory)
            # Hard temp
            g.append(P_batt_kW); lbg.append([self.P_batt_min]); ubg.append([self.P_batt_max])
            g.append(X[0, k]); lbg.append([self.T_mins[0]]); ubg.append([self.T_maxs[0]])

        obj += self.alpha * (X[0, self.N] - self.T_des)**2
        
        # --- SAFETY BOX BOUNDS ---
        # Temp: -10 to 80 (Physics break down outside this)
        x_min_safe = [-10.0, -10.0, -10.0]
        x_max_safe = [80.0, 80.0, 80.0]
        lbx_X = np.tile(x_min_safe, self.N + 1)
        ubx_X = np.tile(x_max_safe, self.N + 1)
        
        lbx_U = np.tile(self.w_mins, self.N)
        ubx_U = np.tile(self.w_maxs, self.N)
        
        self.lbx = np.concatenate([lbx_X, lbx_U])
        self.ubx = np.concatenate([ubx_X, ubx_U])
        self.lbg = np.concatenate(lbg); self.ubg = np.concatenate(ubg)

        OPT_vars = ca.vertcat(ca.vec(X), ca.vec(U))
        OPT_params = ca.vertcat(P_x0, ca.vec(P_dist_expected))
        
        nlp = {'f': obj, 'x': OPT_vars, 'g': ca.vertcat(*g), 'p': OPT_params}
        opts = {
            'ipopt.print_level': 0, # Change from 0 or 1 to 5 for maximum detail
            'print_time': 0, 
            'ipopt.tol': 1e-4, 
            'expand': True,
            'ipopt.bound_mult_init_method': 'mu-based'
        }
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
        x0_val = np.concatenate([self.x_guess, self.u_guess])
        
        try:
            res = self.solver(x0=x0_val, p=p_val, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
            stats = self.solver.stats()

            x_opt = None
            if 'x' in res:
                x_opt = res['x'].full().flatten()

            if stats['success'] or x_opt is not None:
                if not stats['success']:
                    print(f"Warning: {type(self).__name__} using suboptimal solution at step {self.current_step_idx}")
                    print("return_status:", stats.get('return_status', 'unknown'))

                n_x_vars = self.n_x
                n_u_vars = self.n_u

                x_traj = x_opt[0:n_x_vars].reshape((3, self.N + 1), order='F')
                u_traj = x_opt[n_x_vars:n_x_vars + n_u_vars].reshape((2, self.N), order='F')

                self.x_guess = np.hstack([x_traj[:, 1:], x_traj[:, -1:]]).flatten(order='F')
                self.u_guess = np.hstack([u_traj[:, 1:], u_traj[:, -1:]]).flatten(order='F')

                u_opt = u_traj[:, 0]

            else:
                print(f"ERROR: {type(self).__name__} no primal solution at step {self.current_step_idx}")
                u_opt = self.u_guess[:2] if self.u_guess.size >= 2 else np.array([5000.0, 5000.0])

            
        except Exception as e:
            print(f"ERROR: {type(self).__name__} solver CRASHED at step {self.current_step_idx} with error: {e}")
            # Fallback
            self.x_guess = np.zeros(self.n_x)
            self.u_guess = np.zeros(self.n_u)
            u_opt = np.array([5000.0, 5000.0])

        self.current_step_idx += 1 
        return u_opt