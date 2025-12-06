import casadi as ca
import numpy as np
from abc import ABC, abstractmethod
from casadi_ode_solver import rk4_step_ca
from sys_dynamics_casadi import SystemParameters
# from markov_chain import compute_markov_chain # Comentado si no se usa

class BaseController(ABC):
    @abstractmethod
    def compute_control(self, state, disturbance):
        pass

class Thermostat(BaseController):
    '''Rule based controller; threshold based'''
    def __init__(self):
        self.cooling_active = False

    def compute_control(self, state, disturbance):
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
    # SOC_{min}    \le SOC(k)      \le SOC_{max}


class NMPC(BaseController):
    def __init__(self, driving_data, dt=1.0, horizon=10):
        self.dt = dt
        self.N = horizon
        self.params = SystemParameters()
        self.driving_data = driving_data 
        
        self.current_step_idx = 0
        
        # Memory for warm start (Ahora incluye Slacks)
        self.n_x = 3 * (self.N + 1)
        self.n_u = 2 * self.N
        self.n_slack = 2 * (self.N + 1) # 2 variables de holgura por paso (T_min y T_max)

        # --- Constraints Config ---
        # Temperaturas (Se convertirán en Soft Constraints)
        self.T_mins = np.array([30.0, 28.0]) 
        self.T_maxs = np.array([35.0, 34.0])
        
        # Actuadores (Hard Constraints - Límites físicos reales)
        self.w_mins = np.array([0.0, 0.0])   
        self.w_maxs = np.array([10000.0, 10000.0])
        
        self.P_batt_min = -200.0
        self.P_batt_max = 200.0 

        # Cost parameters
        self.alpha = 50.0
        self.T_des = 32.5
        self.rho_soft = 1e5 # Penalización por violar temperatura

        # --- Build Solver ---
        print("Compiling NMPC Solver...")
        self._build_solver()
        print("Done.")

        
        self.x_guess = np.zeros(self.n_x)
        self.u_guess = np.zeros(self.n_u)
        self.slack_guess = np.zeros(self.n_slack)

    def _build_solver(self):
        # 1. Variables Simbólicas
        X = ca.MX.sym('X', 3, self.N + 1) # [T_batt, T_clnt, SOC]
        U = ca.MX.sym('U', 2, self.N)     # [w_comp, w_pump]
        
        # S[0, k]: Cuánto nos pasamos del T_max
        # S[1, k]: Cuánto nos bajamos del T_min
        S = ca.MX.sym('S', 2, self.N + 1) 
        
        P_x0 = ca.MX.sym('P_x0', 3)       
        P_dist = ca.MX.sym('P_dist', 2, self.N) 
        
        obj = 0
        g = []
        lbg = []
        ubg = []
        
        zeros_3 = np.zeros(3)

        # 2. Restricción Inicial
        g.append(X[:, 0] - P_x0)
        lbg.append(zeros_3)
        ubg.append(zeros_3)

        # 3. Bucle del Horizonte
        for k in range(self.N):
            x_next, diag = rk4_step_ca(X[:, k], U[:, k], P_dist[:, k], self.params, self.dt)
            
            # Dinámica
            g.append(x_next - X[:, k+1])
            lbg.append(zeros_3)
            ubg.append(zeros_3)
            
            P_batt_kW = diag[1] / 1000.0 # Ajusta índice según tu rk4
            P_comp_kW = diag[8] / 1000.0 # Ajusta índice según tu rk4
            
            # --- FUNCIÓN DE COSTO ---
            # 1. Energía
            obj += (P_batt_kW + P_comp_kW) * self.dt
            
            # 2. Penalización Soft Constraint (Holguras al cuadrado)
            # Esto evita el crash: Si T > 35, S > 0, el costo sube mucho, pero es factible.
            obj += self.rho_soft * (S[0, k]**2 + S[1, k]**2)
            
            # --- RESTRICCIONES ---
            
            # Potencia (Hard Constraint)
            g.append(P_batt_kW)
            lbg.append([self.P_batt_min])
            ubg.append([self.P_batt_max])
            
            # SOFT CONSTRAINT
            #      X[0] - S[0] <= 35.0
            g.append(X[0, k] - S[0, k])
            ubg.append([self.T_maxs[0]]) 
            lbg.append([-ca.inf])
            
            #      X[0] + S[1] >= 30.0
            g.append(X[0, k] + S[1, k])
            lbg.append([self.T_mins[0]])
            ubg.append([ca.inf])

        # 4. Costo Terminal
        obj += self.alpha * (X[0, self.N] - self.T_des)**2
        
        # 5. Bounds (Límites de variables de decisión)
        
        # T_batt, T_clnt: Los ponemos "infinitos" en lbx/ubx porque la restricción real
        # está manejada arriba en 'g' con las slacks.
        lbx_X = np.tile([-ca.inf, -ca.inf, -ca.inf], self.N + 1)
        ubx_X = np.tile([ca.inf, ca.inf, ca.inf], self.N + 1)
        
        # Actuadores (Hard Bounds)
        lbx_U = np.tile(self.w_mins, self.N)
        ubx_U = np.tile(self.w_maxs, self.N)
        
        # Slacks Bounds: Deben ser positivas (0 a Infinito)
        # S = 0 significa que se cumple la restricción.
        lbx_S = np.zeros(self.n_slack)
        ubx_S = np.full(self.n_slack, ca.inf)
        
        self.lbx = np.concatenate([lbx_X, lbx_U, lbx_S])
        self.ubx = np.concatenate([ubx_X, ubx_U, ubx_S])
        
        self.lbg = np.concatenate(lbg)
        self.ubg = np.concatenate(ubg)

        # 6. NLP Solver
        # Agregamos S al vector de decisión
        OPT_vars = ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(S))
        OPT_params = ca.vertcat(P_x0, ca.vec(P_dist))
        
        nlp = {'f': obj, 'x': OPT_vars, 'g': ca.vertcat(*g), 'p': OPT_params}
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.max_iter': 150, 
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2, # Tolerancia más relajada para casos difíciles
            'ipopt.acceptable_iter': 15,
            'ipopt.warm_start_init_point': 'yes',
            'expand': True
        }
        self.solver = ca.nlpsol('NMPC', 'ipopt', nlp, opts)

    def compute_control(self, state, current_disturbance):
        idx = self.current_step_idx
        len_d = len(self.driving_data)
        end_idx = idx + self.N
        
        # 1. Padding
        if end_idx <= len_d:
            p_driv_horizon = self.driving_data[idx : end_idx]
        else:
            remaining = self.driving_data[idx:]
            p_driv_horizon = np.pad(remaining, (0, self.N - len(remaining)), 'edge')
            
        t_amb_horizon = np.full(self.N, current_disturbance[1])
        d_horizon = np.vstack([p_driv_horizon, t_amb_horizon])
        d_flat = d_horizon.flatten(order='F')
        
        # 2. Setup Solver Inputs
        p_val = np.concatenate([state, d_flat])
        
        # Warm start incluye slacks ahora
        x0_val = np.concatenate([self.x_guess, self.u_guess, self.slack_guess])
        
        try:
            res = self.solver(
                x0=x0_val,
                p=p_val,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg
            )
            
            # Extracción optimizada
            opt_var = res['x'].full().flatten()
            
            # --- Indices ---
            idx_x_end = self.n_x
            idx_u_end = self.n_x + self.n_u
            
            # Control óptimo actual
            u_opt = opt_var[idx_x_end : idx_x_end + 2]
            
            # --- Actualizar Warm Start ---
            
            # Estados
            x_traj = opt_var[:idx_x_end].reshape(3, self.N+1)
            self.x_guess = np.hstack([x_traj[:, 1:], x_traj[:, -1:]]).flatten()
            
            # Controles
            u_traj = opt_var[idx_x_end : idx_u_end].reshape(2, self.N)
            self.u_guess = np.hstack([u_traj[:, 1:], u_traj[:, -1:]]).flatten()
            
            # Slacks
            s_traj = opt_var[idx_u_end:].reshape(2, self.N+1)
            self.slack_guess = np.hstack([s_traj[:, 1:], s_traj[:, -1:]]).flatten()
            
            self.current_step_idx += 1
            return u_opt
            
        except Exception as e:
            print(f'Crash en Solver step {idx}: {e}')
            self.current_step_idx += 1
            self.x_guess = np.zeros(self.n_x)
            self.u_guess = np.zeros(self.n_u)
            self.slack_guess = np.zeros(self.n_slack)
            return np.array([5000.0, 5000.0]) # Max cooling fallback



