import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei"], 
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "figure.figsize": (4.0, 10.0), 
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300
})

import matplotlib.pyplot as plt
import numpy as np

def plot_energy_and_time(
    labels,
    energy_consumption,
    time_computation,
    baseline_label='DP' 
):
    fig, ax = plt.subplots(figsize=(7, 4)) 

    bars = ax.bar(labels, energy_consumption, color='lightgreen', edgecolor='black')
    ax.set_ylabel("Consumo de energía [kJ]")
    ax.set_title("Consumo de Energía Relativo")
    baseline_index = labels.index(baseline_label)
    baseline_energy = energy_consumption[baseline_index]


    for i, bar in enumerate(bars):
        energy_value = bar.get_height()
        
        percentage = (energy_value / baseline_energy) * 100
        
        label_text = f'{percentage:.0f}%'
        
        ax.text(
            bar.get_x() + bar.get_width() / 2, # X Position (center of bar)
            bar.get_height() / 2,              # Y Position (half of height)
            label_text,                        
            ha='center',                       
            va='center',                       
            color='black',                     
        )


    plt.tight_layout()
    plt.show()

    causal_labels = []
    causal_times = []
    nc_labels = []

    for label, t in zip(labels, time_computation):
        if isinstance(t, str):  # e.g. "NC"
            nc_labels.append(label)
        else:
            causal_labels.append(label)
            causal_times.append(t)

    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.bar(causal_labels, causal_times, color='lightgray', edgecolor='black')
    ax.set_ylabel("Tiempo de cómputo [s]")
    ax.set_yscale("log") 
    ax.set_title("Tiempo de Cómputo")

    plt.tight_layout()
    plt.show()


labels = ['DP', 'SMPC', 'DMPC', 'SAC','l-SAC', 'Termostato'] 
energy_consumption = [150, 220, 228, 251, 379, 338]       # kJ
time_computation = ['NC', 0.03, 0.04, 2.8e-5, 2.8e-5, 1e-5] # Seconds


plot_energy_and_time(labels, energy_consumption, time_computation)