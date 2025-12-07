import jax
import jax.numpy as jnp
from efficiency_jax import (get_volumetric_eff, get_isentropic_eff, get_motor_eff,
                            get_pump_pressure_drop, PUMP_MAX_SPEED_RPM, COMP_MAX_SPEED_RPM)
from battery_models_jax import get_ocv_jax, get_rbatt_jax, get_cnom_jax, get_dvdt_jax

# ===============================================================
# 1. PURE PHYSICS ODE (Stateless)
# ===============================================================
def battery_dynamics_ode(state, controls, disturbances, params):
    """
    Calculates dy/dt. All inputs must be JAX arrays or scalars.
    params: SystemParameters object (JAX Pytree)
    """
    # --- Unpack State & Inputs ---
    T_batt, T_clnt, soc = state[0], state[1], state[2]
    w_comp, w_pump = controls[0], controls[1]
    P_driv, T_amb = disturbances[0], disturbances[1]
    
    # --- A. Cooling System Model ---
    eta_vol_pump = get_volumetric_eff(w_pump, PUMP_MAX_SPEED_RPM, 0.98)
    m_clnt_dot_calc = params.V_pump * (w_pump / 60.0) * eta_vol_pump * params.rho_clnt
    m_clnt_dot = jnp.maximum(m_clnt_dot_calc, 0.0)

    delta_p_pump = get_pump_pressure_drop(m_clnt_dot)
    eta_p_motor = get_motor_eff(w_pump)
    
    # Safe division using jnp.where
    P_pump_mech = jnp.where(params.rho_clnt > 0, (m_clnt_dot * delta_p_pump) / params.rho_clnt, 0.0)
    P_pump_elec = jnp.where(eta_p_motor > 0, P_pump_mech / eta_p_motor, 0.0)

    eta_vol_comp = get_volumetric_eff(w_comp, COMP_MAX_SPEED_RPM, 0.95)
    m_rfg_dot = params.V_comp * (w_comp / 60.0) * eta_vol_comp * params.rho_rfg
    
    eta_isen = get_isentropic_eff(w_comp)
    eta_c_motor = get_motor_eff(w_comp)
    h_delta_J = (params.h_cout_kJ - params.h_evaout_kJ) * 1000.0
    
    P_comp_mech = jnp.where(eta_isen > 0, (m_rfg_dot * h_delta_J) / eta_isen, 0.0)
    P_comp_elec = jnp.where(eta_c_motor > 0, P_comp_mech / eta_c_motor, 0.0)
    
    P_cooling = P_pump_elec + P_comp_elec

    # --- B. Electrical Model ---
    P_aux = 200.0
    P_batt_total = P_driv + P_cooling + P_aux

    # JAX Models
    V_oc_cell = get_ocv_jax(soc, T_batt, P_batt_total)
    R_batt_cell = get_rbatt_jax(soc, T_batt)

    # Scale to Pack
    V_oc_pack = V_oc_cell * params.N_series
    R_batt_pack = (R_batt_cell * params.N_series) / params.N_parallel

    # Limits
    C_nom_cell = get_cnom_jax(T_batt)
    C_nom_pack = C_nom_cell * params.N_parallel
    I_max_discharge = 5.0 * C_nom_pack
    I_max_charge = 2.0 * C_nom_pack

    # Current Calculation
    discriminant = V_oc_pack**2 - 4 * R_batt_pack * P_batt_total
    
    # Quadratic Logic
    valid_quadratic = (R_batt_pack > 0) & (discriminant >= 0)
    I_quadratic = (V_oc_pack - jnp.sqrt(jnp.maximum(discriminant, 0.0))) / (2 * R_batt_pack)
    I_linear = jnp.where(V_oc_pack > 0, P_batt_total / V_oc_pack, 0.0)
    
    I_batt = jnp.where(valid_quadratic, I_quadratic, I_linear)

    # SOC Limits (Soft logic for JAX)
    # If SOC > 0.995 AND I < 0 (charging), set I=0
    I_batt = jnp.where((soc >= 0.995) & (I_batt < 0), 0.0, I_batt)
    # If SOC < 0.005 AND I > 0 (discharging), set I=0
    I_batt = jnp.where((soc <= 0.005) & (I_batt > 0), 0.0, I_batt)

    I_batt = jnp.clip(I_batt, -I_max_charge, I_max_discharge)

    # --- C. Thermal Generation ---
    dVdT_cell = get_dvdt_jax(soc)
    dVdT_pack = dVdT_cell * params.N_series
    T_batt_kelvin = T_batt + 273.15
    
    Q_gen = (I_batt**2 * R_batt_pack) - (I_batt * T_batt_kelvin * dVdT_pack)

    # --- D. Heat Transfer (Evaporator) ---
    T_rfg_in = 1.2
    
    # Check flow existence
    has_flow = (m_clnt_dot > 1e-6) & (m_rfg_dot > 1e-6)
    
    # Calculations (executed regardless, masked at end)
    C_clnt_dot = m_clnt_dot * params.C_clnt
    C_rfg_dot = m_rfg_dot * params.C_rfg
    C_min = jnp.minimum(C_clnt_dot, C_rfg_dot)
    C_max = jnp.maximum(C_clnt_dot, C_rfg_dot)
    Cr = C_min / C_max
    UA = params.h_eva * params.A_eva
    NTU = jnp.where(C_min > 0, UA / C_min, 0.0)
    effectiveness = (1.0 - jnp.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)
    Q_max_eva = C_min * (T_clnt - T_rfg_in)
    Q_actual_eva = effectiveness * Q_max_eva
    
    T_clnt_chilled_calc = T_clnt - (Q_actual_eva / C_clnt_dot)
    
    # Apply Logic
    T_clnt_chilled = jnp.where(has_flow, T_clnt_chilled_calc, T_clnt)

    # --- E. Heat Transfer (Battery Cooling) ---
    has_clnt_flow = m_clnt_dot > 1e-6
    exponent = -(params.h_batt * params.A_batt) / (m_clnt_dot * params.C_clnt) # div by zero handled by logic below
    T_clnt_out_calc = T_batt - (T_batt - T_clnt_chilled) * jnp.exp(exponent)
    Q_cool_calc = m_clnt_dot * params.C_clnt * (T_clnt_out_calc - T_clnt_chilled)
    
    Q_cool = jnp.where(has_clnt_flow, Q_cool_calc, 0.0)
    # T_clnt_hot not needed for derivative, just for next step

    # --- F. Derivatives ---
    dT_batt_dt = (Q_gen - Q_cool) / (params.m_batt * params.C_batt)
    
    heat_gain_clnt = Q_cool
    heat_loss_clnt = m_clnt_dot * params.C_clnt * (T_clnt - T_clnt_chilled)
    dT_clnt_dt = (heat_gain_clnt - heat_loss_clnt) / (params.m_clnt_total * params.C_clnt)
    
    Qn_As = C_nom_pack * 3600.0
    dSOC_dt = jnp.where(Qn_As > 0, -I_batt / Qn_As, 0.0)

    # Diagnostic vector (Optional: returned as auxiliary data)
    diagnostics = jnp.array([
        P_cooling, P_batt_total, V_oc_pack, I_batt, Q_gen, Q_cool, m_clnt_dot, P_comp_elec # Last two for the cost function
    ])

    return jnp.array([dT_batt_dt, dT_clnt_dt, dSOC_dt]), diagnostics

# ===============================================================
# 2. RUNGE-KUTTA 4 INTEGRATOR
# ===============================================================
@jax.jit
def rk4_step(state, controls, disturbances, params_tuple, dt):
    """
    Performs one step of RK4 integration.
    """
    # k1
    k1, diag = battery_dynamics_ode(state, controls, disturbances, params_tuple)
    
    # k2
    k2, _ = battery_dynamics_ode(state + 0.5 * dt * k1, controls, disturbances, params_tuple)
    
    # k3
    k3, _ = battery_dynamics_ode(state + 0.5 * dt * k2, controls, disturbances, params_tuple)
    
    # k4
    k4, _ = battery_dynamics_ode(state + dt * k3, controls, disturbances, params_tuple)
    
    # Update
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Return next state AND diagnostics from the START of the step
    return next_state, diag