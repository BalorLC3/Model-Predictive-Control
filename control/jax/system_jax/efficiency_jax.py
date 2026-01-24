import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ===============================================================
# GENERIC COOLING SYSTEM COMPONENT MODELS
# ===============================================================

# Parámetros del Compresor
COMP_MAX_SPEED_RPM = 10000.0
COMP_NOMINAL_SPEED_RPM = 6000.0  # Velocidad de máxima eficiencia
COMP_MAX_VOL_EFF = 0.95         # Eficiencia volumétrica a baja velocidad
COMP_MAX_ISEN_EFF = 0.80        # Eficiencia isentrópica máxima

# Parámetros de la Bomba
PUMP_MAX_SPEED_RPM = 8000.0
PUMP_MAX_VOL_EFF = 0.98
PUMP_PRESSURE_COEFF = 3300.0  # Pa / (kg/s)^2

# Parámetros del Motor Eléctrico (asumido igual para ambos)
MOTOR_MAX_EFF = 0.92
MOTOR_NOMINAL_SPEED_RPM = 5000.0


def get_volumetric_eff(speed_rpm, max_speed_rpm=COMP_MAX_SPEED_RPM, max_eff=COMP_MAX_VOL_EFF):
    s = jnp.maximum(speed_rpm, 0.0)
    slope = 0.4
    eff = max_eff - slope * (s / max_speed_rpm)
    return jnp.clip(eff, 0.0, max_eff)


def get_isentropic_eff(speed_rpm):
    s = jnp.maximum(speed_rpm, 0.0)
    norm_speed_diff = (s - COMP_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.5
    eff = COMP_MAX_ISEN_EFF - k * (norm_speed_diff ** 2)
    return jnp.clip(eff, 0.0, COMP_MAX_ISEN_EFF)


def get_motor_eff(speed_rpm):
    s = jnp.maximum(speed_rpm, 0.0)
    norm_speed_diff = (s - MOTOR_NOMINAL_SPEED_RPM) / COMP_MAX_SPEED_RPM
    k = 0.4
    eff = MOTOR_MAX_EFF - k * (norm_speed_diff ** 2)
    return jnp.clip(eff, 0.0, MOTOR_MAX_EFF)


def get_pump_pressure_drop(m_clnt_dot):
    m = jnp.maximum(m_clnt_dot, 0.0)
    return PUMP_PRESSURE_COEFF * (m ** 2)

