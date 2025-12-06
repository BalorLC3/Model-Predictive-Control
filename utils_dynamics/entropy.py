import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ===============================================================
# SCRIPT DE PRE-PROCESAMIENTO DE DATOS DE ENTROPÍA
#
# Propósito: Leer el archivo CSV original, limpiarlo, procesarlo y
# guardar los arrays limpios de SOC y dV/dT en un archivo .npz.
# ===============================================================

# --- 1. CONFIGURACIÓN ---
FARADAY_CONSTANT = 96485 
show_plot = True # Poner en True para verificar visualmente los datos

folder_raw_data = r'C:\Users\super\Desktop\Supernatural\TESIS\thermal-management\entropy-27-00364-s001'
folder_processed_data = r'C:\Users\super\Desktop\Vasudeva\Manifold\engineering\predictive-control\stochastic-mpc'
os.makedirs(folder_processed_data, exist_ok=True)

input_file = os.path.join(folder_raw_data, 'entropy_data_ageing.csv')
output_file = os.path.join(folder_processed_data, 'entropy_model_data.npz')

# --- 2. LECTURA Y LIMPIEZA DEL CSV ---
try:
    df_raw = pd.read_csv(input_file, sep='\t', comment='#', header=None, usecols=[0, 1])
    df_raw.columns = ['SOC_percent', 'Entropy_J_molK']
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de entrada en {input_file}")
    exit()

df_raw.dropna(inplace=True)
df_raw['SOC_percent'] = pd.to_numeric(df_raw['SOC_percent'], errors='coerce')
df_raw['Entropy_J_molK'] = pd.to_numeric(df_raw['Entropy_J_molK'], errors='coerce')
df_raw.dropna(inplace=True)
df_raw.sort_values(by='SOC_percent', inplace=True) # Buena práctica para la interpolación

# --- 3. PROCESAMIENTO Y CONVERSIÓN ---
df_processed = pd.DataFrame()
df_processed['SOC_fraction'] = df_raw['SOC_percent'] / 100
df_processed['dVdT_V_K'] = df_raw['Entropy_J_molK'] / FARADAY_CONSTANT

# --- 4. GUARDADO DE LOS ARRAYS FINALES ---
np.savez(output_file, 
         soc_points=df_processed['SOC_fraction'].values, 
         dvdt_points=df_processed['dVdT_V_K'].values)

print(f"Datos de entropía procesados y guardados exitosamente en:\n{output_file}")

# --- 5. VISUALIZACIÓN OPCIONAL ---
if show_plot:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_processed['SOC_fraction'] * 100, df_processed['dVdT_V_K'] * 1000, 'r-', marker='.', label='dV/dT')
    ax.set_title('Coeficiente de Calor Entrópico (dV/dT) vs. SOC')
    ax.set_xlabel('Estado de Carga (%)')
    ax.set_ylabel('dV/dT (mV/K)')
    ax.grid(True)
    ax.legend()
    plt.show()