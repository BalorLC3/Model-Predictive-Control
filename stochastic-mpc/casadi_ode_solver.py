import casadi as ca
import numpy as np

# Importamos las versiones "híbridas" (CasADi-safe) que creamos antes
from efficiency import (get_volumetric_eff, get_isentropic_eff, get_motor_eff,
                        get_pump_pressure_drop, PUMP_MAX_SPEED_RPM, COMP_MAX_SPEED_RPM)
from battery_models import get_ocv, get_rbatt, get_cnom, get_dvdt

# ===============================================================
# 1. PURE PHYSICS ODE (Stateless - CasADi Version)
# ===============================================================
def battery_dynamics_ode_ca(state, controls, disturbances, params):
    """
    Calculates dy/dt using CasADi symbolic operations.
    
    Args:
        state: ca.MX/SX or DM vector [T_batt, T_clnt, soc]
        controls: ca.MX/SX or DM vector [w_comp, w_pump]
        disturbances: ca.MX/SX or DM vector [P_driv, T_amb]
        params: SystemParameters object (instance)
    """
    # --- Unpack State & Inputs ---
    T_batt, T_clnt, soc = state[0], state[1], state[2]
    w_comp, w_pump = controls[0], controls[1]
    P_driv, T_amb = disturbances[0], disturbances[1]
    
    # --- A. Cooling System Model ---
    eta_vol_pump = get_volumetric_eff(w_pump, PUMP_MAX_SPEED_RPM, 0.98)
    m_clnt_dot_calc = params.V_pump * (w_pump / 60.0) * eta_vol_pump * params.rho_clnt
    
    # Soft Maximum para evitar discontinuidades abruptas en MPC (opcional pero recomendado)
    # ca.fmax es el equivalente a jnp.maximum
    m_clnt_dot = ca.fmax(m_clnt_dot_calc, 0.0)

    delta_p_pump = get_pump_pressure_drop(m_clnt_dot)
    eta_p_motor = get_motor_eff(w_pump)
    
    # Safe division using ca.if_else
    # Si rho > 0 (que siempre es cierto en params), calculamos.
    P_pump_mech = (m_clnt_dot * delta_p_pump) / params.rho_clnt
    
    # Protección contra división por cero en eficiencia
    P_pump_elec = P_pump_mech / ca.fmax(eta_p_motor, 0.01)

    eta_vol_comp = get_volumetric_eff(w_comp, COMP_MAX_SPEED_RPM, 0.95)
    m_rfg_dot = params.V_comp * (w_comp / 60.0) * eta_vol_comp * params.rho_rfg
    
    eta_isen = get_isentropic_eff(w_comp)
    eta_c_motor = get_motor_eff(w_comp)
    h_delta_J = (params.h_cout_kJ - params.h_evaout_kJ) * 1000.0
    
    P_comp_mech = (m_rfg_dot * h_delta_J) / ca.fmax(eta_isen, 0.01)
    P_comp_elec = P_comp_mech / ca.fmax(eta_c_motor, 0.01)
    
    # Lógica de apagado estricto (para evitar consumo fantasma cuando w < 10)
    P_pump_elec = ca.if_else(w_pump < 10.0, 0.0, P_pump_elec)
    P_comp_elec = ca.if_else(w_comp < 10.0, 0.0, P_comp_elec)
    
    P_cooling = P_pump_elec + P_comp_elec

    # --- B. Electrical Model ---
    P_aux = 200.0
    P_batt_total = P_driv + P_cooling + P_aux

    # CasADi Models (importados de battery_models.py)
    # Asumimos que get_ocv maneja internamente la lógica o usamos 'discharge' por defecto
    # Para ser robustos en CasADi, mejor no usar strings condicionales dentro de la función si es posible
    V_oc_cell = get_ocv(soc, T_batt, mode='discharge') 
    R_batt_cell = get_rbatt(soc, T_batt)

    # Scale to Pack
    V_oc_pack = V_oc_cell * params.N_series
    R_batt_pack = (R_batt_cell * params.N_series) / params.N_parallel

    # Limits
    C_nom_cell = get_cnom(T_batt)
    C_nom_pack = C_nom_cell * params.N_parallel
    I_max_discharge = 5.0 * C_nom_pack
    I_max_charge = 2.0 * C_nom_pack

    # Current Calculation
    discriminant = V_oc_pack**2 - 4 * R_batt_pack * P_batt_total
    
    # Quadratic Logic
    # ca.fmax(disc, 0) evita NaNs en la raíz cuadrada
    sqrt_disc = ca.sqrt(ca.fmax(discriminant, 0.0))
    I_quadratic = (V_oc_pack - sqrt_disc) / (2 * R_batt_pack)
    I_linear = P_batt_total / V_oc_pack # Fallback simple
    
    # Si disc >= 0 usamos cuadrática, sino lineal
    I_batt = ca.if_else(discriminant >= 0, I_quadratic, I_linear)

    # Saturación de corriente (Clip)
    I_batt = ca.fmin(ca.fmax(I_batt, -I_max_charge), I_max_discharge)
    
    # SOC Limits (Logic)
    # ca.if_else(condition, true_val, false_val)
    # Si SOC > 0.995 Y Corriente < 0 (Carga) -> I=0
    I_batt = ca.if_else(ca.logic_and(soc >= 0.995, I_batt < 0), 0.0, I_batt)
    # Si SOC < 0.005 Y Corriente > 0 (Descarga) -> I=0
    I_batt = ca.if_else(ca.logic_and(soc <= 0.005, I_batt > 0), 0.0, I_batt)

    # --- C. Thermal Generation ---
    dVdT_cell = get_dvdt(soc)
    dVdT_pack = dVdT_cell * params.N_series
    T_batt_kelvin = T_batt + 273.15
    
    Q_gen = (I_batt**2 * R_batt_pack) - (I_batt * T_batt_kelvin * dVdT_pack)

    # --- D. Heat Transfer (Evaporator) ---
    T_rfg_in = 1.2
    
    # Flow existence check logic
    has_flow = ca.logic_and(m_clnt_dot > 1e-6, m_rfg_dot > 1e-6)
    
    C_clnt_dot = m_clnt_dot * params.C_clnt
    C_rfg_dot = m_rfg_dot * params.C_rfg
    C_min = ca.fmin(C_clnt_dot, C_rfg_dot)
    C_max = ca.fmax(C_clnt_dot, C_rfg_dot)
    Cr = C_min / (C_max + 1e-9) # Epsilon para evitar div/0
    
    UA = params.h_eva * params.A_eva
    NTU = ca.if_else(C_min > 0, UA / C_min, 0.0)
    
    effectiveness = (1.0 - ca.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)
    Q_max_eva = C_min * (T_clnt - T_rfg_in)
    Q_actual_eva = effectiveness * Q_max_eva
    
    T_clnt_chilled_calc = T_clnt - (Q_actual_eva / (C_clnt_dot + 1e-9))
    
    T_clnt_chilled = ca.if_else(has_flow, T_clnt_chilled_calc, T_clnt)

    # --- E. Heat Transfer (Battery Cooling) ---
    has_clnt_flow = m_clnt_dot > 1e-6
    exponent = -(params.h_batt * params.A_batt) / (m_clnt_dot * params.C_clnt + 1e-9)
    T_clnt_out_calc = T_batt - (T_batt - T_clnt_chilled) * ca.exp(exponent)
    Q_cool_calc = m_clnt_dot * params.C_clnt * (T_clnt_out_calc - T_clnt_chilled)
    
    Q_cool = ca.if_else(has_clnt_flow, Q_cool_calc, 0.0)

    # --- F. Derivatives ---
    dT_batt_dt = (Q_gen - Q_cool) / (params.m_batt * params.C_batt)
    
    heat_gain_clnt = Q_cool
    heat_loss_clnt = m_clnt_dot * params.C_clnt * (T_clnt - T_clnt_chilled)
    dT_clnt_dt = (heat_gain_clnt - heat_loss_clnt) / (params.m_clnt_total * params.C_clnt)
    
    Qn_As = C_nom_pack * 3600.0
    dSOC_dt = -I_batt / Qn_As

    # Diagnostic vector (Vertcat para crear vector columna CasADi)
    diagnostics = ca.vertcat(
        P_cooling, P_batt_total, V_oc_pack, I_batt, Q_gen, Q_cool, m_clnt_dot, T_clnt_chilled, P_comp_elec
    )

    return ca.vertcat(dT_batt_dt, dT_clnt_dt, dSOC_dt), diagnostics


# ===============================================================
# 2. RUNGE-KUTTA 4 INTEGRATOR (CasADi Version)
# ===============================================================
def rk4_step_ca(state, controls, disturbances, params, dt):
    """
    Performs one step of RK4 integration using CasADi symbolic graphs.
    Returns symbolic expression for next state.
    """
    # k1
    k1, diag = battery_dynamics_ode_ca(state, controls, disturbances, params)
    
    # k2
    k2, _ = battery_dynamics_ode_ca(state + 0.5 * dt * k1, controls, disturbances, params)
    
    # k3
    k3, _ = battery_dynamics_ode_ca(state + 0.5 * dt * k2, controls, disturbances, params)
    
    # k4
    k4, _ = battery_dynamics_ode_ca(state + dt * k3, controls, disturbances, params)
    
    # Update
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Return next state AND diagnostics from the START of the step
    return next_state, diag