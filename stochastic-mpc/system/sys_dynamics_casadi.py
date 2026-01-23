import casadi as ca
import numpy as np
from system.casadi_ode_solver import rk4_step_ca

class SystemParameters:
    def __init__(self):
        '''Initialize system parameters'''
        # --- Thermodynamics ---
        self.rho_rfg = 27.8
        self.rho_clnt = 1069.5
        self.C_rfg = 1117.0
        self.C_clnt = 3330.0
        self.V_comp = 33e-6
        self.V_pump = 33e-6
        self.h_eva = 1000.0
        self.A_eva = 0.3
        self.h_batt = 300.0
        self.A_batt = 1.0
        self.PR = 5.0
        self.h_cout_kJ = 284.3
        self.h_evaout_kJ = 250.9
        
        # --- Battery pack parameters ---
        self.m_batt = 40.0       
        self.C_batt = 1350.0     
        self.N_series = 96.0       
        self.N_parallel = 1.0     
        
        self.m_clnt_total = 2.0 * self.rho_clnt / 1000

class BatteryThermalSystem:
    def __init__(self, initial_state, params):
        '''Initialize the battery thermal system'''
        self.params = params
        
        # In CasADi, is more efficient to use numpy arrays for state
        self.state = np.array([
            initial_state['T_batt'],
            initial_state['T_clnt'],
            initial_state['soc']
        ])
        
        self.diagnostics = {}
        
        # Compile an integrator
        # This creates a CasADi function that can be called repeatedly 
        self._build_step_function()

    def _build_step_function(self):
        """
        Constructs the CasADi function for system dynamics integration.
        """
        # Deine symbolic variables (Placeholders)
        x_sym = ca.MX.sym('x', 3)   # [T_batt, T_clnt, soc]
        u_sym = ca.MX.sym('u', 2)   # [w_comp, w_pump]
        d_sym = ca.MX.sym('d', 2)   # [P_driv, T_amb]
        dt_sym = ca.MX.sym('dt')    # Timestep
        
        # Call the physics (RK4) using symbols
        # Here we pass 'self.params' as an object. The constants are "printed" in the graph.
        x_next_sym, diag_sym = rk4_step_ca(x_sym, u_sym, d_sym, self.params, dt_sym)
        
        # Create the Compiled Function
        # Inputs: [State, Control, Disturbance, DT]
        # Outputs: [Siguiente Estado, Diagnósticos]
        self.integrator_fn = ca.Function(
            'sys_step', 
            [x_sym, u_sym, d_sym, dt_sym], 
            [x_next_sym, diag_sym],
            ['x', 'u', 'd', 'dt'], 
            ['x_next', 'diag']
        )

    def step(self, controls, disturbances, dt):
        """
        Avanza la simulación un paso usando el integrador CasADi compilado.
        """
        # 1. Execute the compiled function
        # CasADi accepts lists or numpy arrays automatically
        res = self.integrator_fn(self.state, controls, disturbances, dt)
        
        # 2. Extract results (CasADi returns DM matrices, convert to numpy flatten)
        # res[0] is x_next
        # res[1] is diag
        self.state = np.array(res[0]).flatten()
        diag_vec = np.array(res[1]).flatten()
        
        # Package diagnostics into a dictionary
        # The order must match the return of 'battery_dynamics_ode_ca' in casadi_ode_solver.py
        # [P_cooling, P_batt_total, V_oc, I_batt, Q_gen, Q_cool, m_clnt_dot, T_chilled]
        self.diagnostics = {
            'P_cooling': diag_vec[0],
            'P_batt_total': diag_vec[1],
            'V_oc_pack': diag_vec[2],
            'I_batt': diag_vec[3],
            'Q_gen': diag_vec[4],
            'Q_cool': diag_vec[5],
            'm_clnt_dot': diag_vec[6],
            'T_chilled': diag_vec[7]
        }
        
        return self.state, self.diagnostics