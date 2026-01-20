import numpy as np
import casadi as ca

# ===============================================================
# MoDULO DE MODELOS DE BATERiA 
# ===============================================================

# --- PARÁMETROS GLOBALES ---

# OCV Data
ocv_temperatures = np.array([5, 15, 25, 45])
ocv_params_charge = np.array([
    [3.734, -0.2756, 8.013e-5, -0.001155, -0.06674],
    [3.629, -0.3191, 6.952e-5, -0.0005411, -0.1493],
    [3.637, -0.3091, 7.033e-5, -0.0005477, -0.1366],
    [3.584, -0.4868, 6.212e-5, -0.0001919, -0.2181],
])
ocv_params_discharge = np.array([
    [3.804, -0.3487, 8.838e-5, -0.001618, -0.04724],
    [3.599, -0.2933, 6.9e-5,   -0.0004687, -0.1434],
    [3.604, -0.2803, 6.871e-5, -0.0004523, -0.1341],
    [3.55,  -0.4573, 6.02e-5,  -6.241e-5,  -0.221],
])

# Resistencia y Capacidad
res_temperatures = np.array([5, 15, 25, 45])
cnom_table = np.array([17.17, 19.24, 20.0, 21.6])
r0_table = np.array([0.007, 0.0047, 0.003, 0.0019])
r1_table = np.array([0.0042, 0.0018, 0.00065, 0.00054])

# Calor Entrópico
soc_grid = np.array([-1.000e-04, 4.990e-02, 9.990e-02, 1.499e-01, 1.999e-01, 2.500e-01,
        3.000e-01, 3.500e-01, 4.000e-01, 4.500e-01, 5.000e-01, 5.500e-01,
        6.000e-01, 6.500e-01, 7.000e-01, 7.500e-01, 8.000e-01, 8.500e-01,
        9.000e-01, 9.500e-01, 1.000e+00])
dvdt_grid = np.array([-5.05259901e-04, -3.86484957e-04, -1.83862780e-04, -1.45203914e-04,
            -1.01362908e-04, -7.95978631e-05, -2.97455572e-05, 6.21858344e-05,
            1.06130487e-04, 1.28724671e-04, 1.39088981e-04, 1.38363481e-04,
            1.33595895e-04, 7.88723628e-05, -3.42022076e-06, -1.48209565e-05,
            -2.12468258e-05, -2.45634037e-05, -3.57568533e-05, -3.33730641e-05,
            -3.50313530e-05])

# --- CREACIÓN DE INTERPOLADORES (GLOBAL - UNA SOLA VEZ) ---
# Esto es crítico para la velocidad. No los crees dentro de las funciones.

# OCV Interpolants (Lista de listas)
# 0: Charge Params Interpolants, 1: Discharge Params Interpolants
interp_ocv_charge = []
interp_ocv_discharge = []

for i in range(5):
    interp_ocv_charge.append(
        ca.interpolant(f'ocv_chg_{i}', 'linear', [ocv_temperatures], ocv_params_charge[:, i])
    )
    interp_ocv_discharge.append(
        ca.interpolant(f'ocv_dis_{i}', 'linear', [ocv_temperatures], ocv_params_discharge[:, i])
    )

# Resistance & Capacity Interpolants
r0_interpolant = ca.interpolant('r0_int', 'linear', [res_temperatures], r0_table)
r1_interpolant = ca.interpolant('r1_int', 'linear', [res_temperatures], r1_table)
cnom_interpolant = ca.interpolant('cnom_int', 'linear', [res_temperatures], cnom_table)
dvdt_interpolant = ca.interpolant('dvdt_int', 'linear', [soc_grid], dvdt_grid)

# --- HELPER PARA HIBRIDACIÓN ---
def _to_numeric_if_needed(val, input_ref):
    """
    Si la entrada (input_ref) es un número (float/int/numpy), convierte el resultado
    de CasADi (DM) a float. Si es simbólico, lo deja como está.
    """
    if isinstance(input_ref, (float, int, np.float64, np.float32)):
        return float(val)
    if isinstance(input_ref, np.ndarray) and input_ref.dtype in [np.float64, np.float32]:
        return float(val) # Asumiendo escalar para interpolación simple
    return val

# ===============================================================
# --- FUNCIONES DEL MODELO ---
# ===============================================================

def get_ocv(soc, temp, mode='discharge'):
    soc_percent = soc * 100

    if mode == 'charge':
        interpolators = interp_ocv_charge
    else:
        interpolators = interp_ocv_discharge
        
    # Obtener parámetros (devuelve DM si temp es float)
    p0 = interpolators[0](temp)
    p1 = interpolators[1](temp)
    p2 = interpolators[2](temp)
    a1 = interpolators[3](temp)
    a2 = interpolators[4](temp)
    
    # Calcular OCV (Operaciones simbólicas o numéricas de CasADi)
    ocv = p0 * ca.exp(a1 * soc_percent) + p1 * ca.exp(a2 * soc_percent) + p2 * soc_percent**2
    
    # Convertir a float si estamos en simulación numérica
    return _to_numeric_if_needed(ocv, temp)


def get_rbatt(soc, temp):
    SCALE_FACTOR = 3.0
    
    r0 = r0_interpolant(temp)
    r1 = r1_interpolant(temp)
    
    r_total_cell = (r0 + r1) * 1.0 # soc_multiplier = 1
    r_total_pack = r_total_cell / SCALE_FACTOR
    
    return _to_numeric_if_needed(r_total_pack, temp)


def get_cnom(temp):
    val = cnom_interpolant(temp)
    return _to_numeric_if_needed(val, temp)


def get_dvdt(soc):
    val = dvdt_interpolant(soc)
    return _to_numeric_if_needed(val, soc)