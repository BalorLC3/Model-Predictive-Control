import casadi as ca
import numpy as np

# ===============================================================
# GENERIC COOLING SYSTEM COMPONENT MODELS (CASADI VERSION)
# ===============================================================

# Compressor parameters
COMP_MAX_SPEED_RPM = 10000.0
COMP_NOMINAL_SPEED_RPM = 6000.0  # Maximum efficiency speed
COMP_MAX_VOL_EFF = 0.95         # Volumetric efficiency at low speed
COMP_MAX_ISEN_EFF = 0.80        # Maximum isentropic efficiency

# Pump parameters
PUMP_MAX_SPEED_RPM = 8000.0
PUMP_MAX_VOL_EFF = 0.98
PUMP_PRESSURE_COEFF = 3300.0  # Pa / (kg/s)^2

# Motor parameters
MOTOR_MAX_EFF = 0.92
MOTOR_NOMINAL_SPEED_RPM = 5000.0

# --- Intern helper to replicate jnp.clip ---
def _clip(val, min_val, max_val):
    """Equivalente a jnp.clip(val, min, max) usando primitivas CasADi"""
    return ca.fmin(ca.fmax(val, min_val), max_val)

def get_volumetric_eff(speed_rpm, max_speed_rpm=COMP_MAX_SPEED_RPM, max_eff=COMP_MAX_VOL_EFF):
    # ca.fmax is equivalent to jnp.maximum
    s = ca.fmax(speed_rpm, 0.0)
    
    slope = 0.4
    # Avoid division by zero if max_speed_rpm were symbolic (though it's constant here)
    eff = max_eff - slope * (s / (max_speed_rpm + 1e-9))
    
    return _clip(eff, 0.0, max_eff)


def get_isentropic_eff(speed_rpm):
    s = ca.fmax(speed_rpm, 0.0)
    
    norm_speed_diff = (s - COMP_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.5
    eff = COMP_MAX_ISEN_EFF - k * (norm_speed_diff ** 2)
    
    return _clip(eff, 0.0, COMP_MAX_ISEN_EFF)


def get_motor_eff(speed_rpm):
    s = ca.fmax(speed_rpm, 0.0)
    
    norm_speed_diff = (s - MOTOR_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.4
    eff = MOTOR_MAX_EFF - k * (norm_speed_diff ** 2)
    
    return _clip(eff, 0.0, MOTOR_MAX_EFF)


def get_pump_pressure_drop(m_clnt_dot):
    m = ca.fmax(m_clnt_dot, 0.0)
    return PUMP_PRESSURE_COEFF * (m ** 2)