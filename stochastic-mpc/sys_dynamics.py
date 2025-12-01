import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. IMPORTACIONES ---
# Asegúrate de que estos archivos (efficiency.py, battery_models.py, controllers.py) 
# estén en la misma carpeta o ajusta el path.
try:
    from efficiency import (get_volumetric_eff, get_isentropic_eff, get_motor_eff,
                            get_pump_pressure_drop, PUMP_MAX_SPEED_RPM, COMP_MAX_SPEED_RPM)
    from battery_models import get_ocv, get_rbatt, get_cnom, get_dvdt
    from controllers import Thermostat
except ImportError:
    print("ADVERTENCIA: No se encontraron los módulos externos (efficiency, battery_models, controllers).")
    print("El script fallará si no se definen estas funciones.")

# --- 2. CLASE DE PARÁMETROS (CORREGIDA) ---
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
        
        # --- PARÁMETROS DEL PACK DE BATERÍA ---
        self.m_batt = 40.0       # Masa total del pack [kg]
        self.C_batt = 1350.0     # Capacidad calorífica [J/kgK]
        self.N_series = 96       # Celdas en serie (Aprox 400V)
        self.N_parallel = 1      # Ramas en paralelo
        
        self.m_clnt_total = 2.0 * self.rho_clnt / 1000


# --- 3. CLASE PRINCIPAL DEL SISTEMA (CORREGIDA) ---
class BatteryThermalSystem:
    def __init__(self, initial_state, params):
        self.params = params
        self.state = np.array([
            initial_state['T_batt'],
            initial_state['T_clnt'],
            initial_state['soc']
        ])
        self.diagnostics = {}

    def _system_dynamics(self, t, y, u, d):
        T_batt, T_clnt, soc = y[0], y[1], y[2]
        w_comp, w_pump = u[0], u[1]
        P_driv, T_amb = d[0], d[1] 

        # --- Modelo del Sistema de Enfriamiento ---
        eta_vol_pump = get_volumetric_eff(w_pump, PUMP_MAX_SPEED_RPM, 0.98)
        m_clnt_dot = self.params.V_pump * (w_pump / 60) * eta_vol_pump * self.params.rho_clnt
        
        # CORRECCIÓN: Permitir flujo cero real, solo evitar negativos
        m_clnt_dot = max(m_clnt_dot, 0.0)

        delta_p_pump = get_pump_pressure_drop(m_clnt_dot)
        eta_p_motor = get_motor_eff(w_pump)
        P_pump_mech = (m_clnt_dot * delta_p_pump) / self.params.rho_clnt if self.params.rho_clnt > 0 else 0
        P_pump_elec = P_pump_mech / eta_p_motor if eta_p_motor > 0 else 0

        eta_vol_comp = get_volumetric_eff(w_comp, COMP_MAX_SPEED_RPM, 0.95)
        m_rfg_dot = self.params.V_comp * (w_comp / 60) * eta_vol_comp * self.params.rho_rfg 
        eta_isen = get_isentropic_eff(w_comp)
        eta_c_motor = get_motor_eff(w_comp)
        h_delta_J = (self.params.h_cout_kJ - self.params.h_evaout_kJ) * 1000
        P_comp_mech = (m_rfg_dot * h_delta_J) / eta_isen if eta_isen > 0 else 0
        P_comp_elec = P_comp_mech / eta_c_motor if eta_c_motor > 0 else 0
        P_cooling = P_pump_elec + P_comp_elec

        # --- Modelo Eléctrico (ESCALADO AL PACK) ---
        P_aux = 200 # W auxiliares
        P_batt_total = P_driv + P_cooling + P_aux 

        ocv_mode = 'charge' if P_batt_total < 0 else 'discharge'
        
        # 1. Obtener valores de celda unitaria
        V_oc_cell = get_ocv(soc, T_batt, mode=ocv_mode)
        R_batt_cell = get_rbatt(soc, T_batt)
        
        # 2. Escalar a Pack Completo
        V_oc_pack = V_oc_cell * self.params.N_series
        R_batt_pack = (R_batt_cell * self.params.N_series) / self.params.N_parallel
        
        # 3. Límites de corriente basados en capacidad del pack
        C_nom_cell = get_cnom(T_batt)
        C_nom_pack = C_nom_cell * self.params.N_parallel
        
        I_max_discharge = 5.0 * C_nom_pack # Permite picos altos
        I_max_charge    = 2.0 * C_nom_pack    

        # 4. Cálculo de corriente (Ecuación cuadrática)
        discriminant = V_oc_pack**2 - 4 * R_batt_pack * P_batt_total
        
        if R_batt_pack > 0 and discriminant >= 0:
            I_batt = (V_oc_pack - np.sqrt(discriminant)) / (2 * R_batt_pack)
        else:
            # Fallback si la potencia demandada excede la capacidad física
            I_batt = P_batt_total / V_oc_pack if V_oc_pack > 0 else 0

        # Bloqueos de seguridad SOC
        if soc >= 0.995 and I_batt < 0: I_batt = 0.0
        if soc <= 0.005 and I_batt > 0: I_batt = 0.0

        I_batt = np.clip(I_batt, -I_max_charge, I_max_discharge)

        # Recalcular potencia real eléctrica
        P_batt_real = V_oc_pack * I_batt # Potencia en bornes (aprox)

        # --- Modelo Térmico (Generación de Calor) ---
        dVdT_cell = get_dvdt(soc)
        dVdT_pack = dVdT_cell * self.params.N_series # Escalar término entrópico
        
        T_batt_kelvin = T_batt + 273.15
        # Calor irreversible (Joule) + Reversible (Entrópico)
        Q_gen = (I_batt**2 * R_batt_pack) - (I_batt * T_batt_kelvin * dVdT_pack)

        # --- Transferencia de Calor ---
        T_clnt_chilled = self._model_evaporator(T_clnt, m_clnt_dot, m_rfg_dot)
        T_clnt_hot, Q_cool = self._model_battery_cooling(T_batt, T_clnt_chilled, m_clnt_dot)

        # --- Derivadas ---
        dT_batt_dt = (Q_gen - Q_cool) / (self.params.m_batt * self.params.C_batt)
        
        heat_gain_clnt = Q_cool
        heat_loss_clnt = m_clnt_dot * self.params.C_clnt * (T_clnt - T_clnt_chilled)
        dT_clnt_dt = (heat_gain_clnt - heat_loss_clnt) / (self.params.m_clnt_total * self.params.C_clnt)
        
        Qn_As = C_nom_pack * 3600
        dSOC_dt = -I_batt / Qn_As if Qn_As > 0 else 0

        # Telemetría
        self.diagnostics = {
            'P_cooling': P_cooling, 'P_batt_total': P_batt_total,
            'V_oc_pack': V_oc_pack, 'I_batt': I_batt,
            'Q_gen': Q_gen, 'Q_cool': Q_cool,
            'm_clnt_dot': m_clnt_dot, 'T_chilled': T_clnt_chilled
        }
        
        return [dT_batt_dt, dT_clnt_dt, dSOC_dt]

    def step(self, controls, disturbances, dt):
        t_span = [0, dt]
        sol = solve_ivp(
            fun=self._system_dynamics, t_span=t_span, y0=self.state,
            args=(controls, disturbances), method='RK45'
        )
        self.state = sol.y[:, -1]
        return self.state, self.diagnostics

    def _model_evaporator(self, T_clnt_in, m_clnt_dot, m_rfg_dot):
        T_rfg_in = 1.2 # Temperatura evaporación refrigerante
        
        # Si no hay flujo, no hay cambio de temperatura
        if m_clnt_dot < 1e-6 or m_rfg_dot < 1e-6:
            return T_clnt_in

        C_clnt_dot = m_clnt_dot * self.params.C_clnt
        C_rfg_dot = m_rfg_dot * self.params.C_rfg
        C_min = min(C_clnt_dot, C_rfg_dot)
        C_max = max(C_clnt_dot, C_rfg_dot)
        Cr = C_min / C_max
        UA = self.params.h_eva * self.params.A_eva
        NTU = UA / C_min
        effectiveness = (1 - np.exp(-NTU * (1 + Cr))) / (1 + Cr)
        Q_max = C_min * (T_clnt_in - T_rfg_in)
        Q_actual = effectiveness * Q_max
        T_clnt_out = T_clnt_in - (Q_actual / C_clnt_dot)
        return T_clnt_out

    def _model_battery_cooling(self, T_batt, T_clnt_in, m_clnt_dot):
        # CORRECCIÓN: Si no hay flujo, Q_cool es 0 exacto
        if m_clnt_dot < 1e-6: 
            return T_clnt_in, 0.0
            
        exponent = -(self.params.h_batt * self.params.A_batt) / (m_clnt_dot * self.params.C_clnt)
        T_clnt_out = T_batt - (T_batt - T_clnt_in) * np.exp(exponent)
        Q_cool = m_clnt_dot * self.params.C_clnt * (T_clnt_out - T_clnt_in)
        return T_clnt_out, Q_cool
