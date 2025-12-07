import casadi as ca
import numpy as np

# Importamos el solver RK4 de CasADi que creamos en el paso anterior
from casadi_ode_solver import rk4_step_ca

class SystemParameters:
    def __init__(self):
        # --- Termodinámica ---
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
        
        # --- Parámetros del Pack ---
        self.m_batt = 40.0       
        self.C_batt = 1350.0     
        self.N_series = 96.0       
        self.N_parallel = 1.0     
        
        self.m_clnt_total = 2.0 * self.rho_clnt / 1000

# Los parámetros se pasan como objetos normales de Python durante la construcción del grafo.

class BatteryThermalSystem:
    def __init__(self, initial_state, params):
        self.params = params
        
        # En CasADi usamos numpy arrays estándar para el estado numérico
        self.state = np.array([
            initial_state['T_batt'],
            initial_state['T_clnt'],
            initial_state['soc']
        ])
        
        self.diagnostics = {}
        
        # --- COMPILAR EL INTEGRADOR (Solo una vez) ---
        # Esto crea una función C++ eficiente en memoria
        self._build_step_function()

    def _build_step_function(self):
        """
        Construye el grafo simbólico de CasADi y crea una ca.Function.
        """
        # 1. Definir variables simbólicas (Placeholders)
        x_sym = ca.MX.sym('x', 3)   # [T_batt, T_clnt, soc]
        u_sym = ca.MX.sym('u', 2)   # [w_comp, w_pump]
        d_sym = ca.MX.sym('d', 2)   # [P_driv, T_amb]
        dt_sym = ca.MX.sym('dt')    # Timestep
        
        # 2. Llamar a la física (RK4) usando símbolos
        # Aquí pasamos 'self.params' como objeto. Las constantes se "imprimen" en el grafo.
        x_next_sym, diag_sym = rk4_step_ca(x_sym, u_sym, d_sym, self.params, dt_sym)
        
        # 3. Crear la Función Compilada
        # Inputs: [Estado, Control, Perturbación, DT]
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
        # 1. Ejecutar la función compilada
        # CasADi acepta listas o numpy arrays automáticamente
        res = self.integrator_fn(self.state, controls, disturbances, dt)
        
        # 2. Extraer resultados (CasADi devuelve matrices DM, convertimos a numpy flatten)
        # res[0] es x_next
        # res[1] es diag
        self.state = np.array(res[0]).flatten()
        diag_vec = np.array(res[1]).flatten()
        
        # 3. Empaquetar diagnósticos para telemetría
        # El orden debe coincidir con el retorno de 'battery_dynamics_ode_ca' en casadi_ode_solver.py
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