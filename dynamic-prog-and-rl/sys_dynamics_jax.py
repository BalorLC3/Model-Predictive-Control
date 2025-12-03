import jax.numpy as jnp
from jax_ode_solver import rk4_step
import jax.tree_util

class SystemParameters:
    def __init__(self):
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
        
        # --- PARÁMETROS DEL PACK  ---
        self.m_batt = 40.0       
        self.C_batt = 1350.0     
        self.N_series = 96.0       
        self.N_parallel = 1.0     
        
        self.m_clnt_total = 2.0 * self.rho_clnt / 1000

def _params_flatten(obj):
    keys = sorted(vars(obj).keys())
    children = [getattr(obj, k) for k in keys]
    aux_data = keys
    return children, aux_data

def _params_unflatten(aux_data, children):
    obj = SystemParameters.__new__(SystemParameters)
    for key, val in zip(aux_data, children):
        setattr(obj, key, val)
    return obj

jax.tree_util.register_pytree_node(
    SystemParameters,
    _params_flatten,
    _params_unflatten
)

class BatteryThermalSystem:
    def __init__(self, initial_state, params):
        self.params = params
        # Convert initial state to JAX array immediately
        self.state = jnp.array([
            initial_state['T_batt'],
            initial_state['T_clnt'],
            initial_state['soc']
        ])
        
        self.diagnostics = {}

    def step(self, controls, disturbances, dt):
        """
        Advances the simulation by one step using RK4 (JAX).
        """
        # Ensure inputs are JAX arrays
        controls_jax = jnp.array(controls)
        disturbances_jax = jnp.array(disturbances)
        
        # --- CALL THE JIT COMPILED SOLVER ---
        next_state, diag_vec = rk4_step(
            self.state, 
            controls_jax, 
            disturbances_jax, 
            self.params, 
            dt
        )
        
        self.state = next_state
        
        # Unpack diagnostics for plotting 
        # Order matches jax_ode_solver.py return
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
