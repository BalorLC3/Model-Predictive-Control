from dataclasses import dataclass
import numpy as np
import pandas as pd
import time
from typing import Any

@dataclass
class SimConfiguration:
    """Holds the scenario parameters to ensure all controllers run on identical conditions"""
    driving_data: np.array
    velocity_data: np.array
    T_amb: float
    dt: float = 1.0
    total_time: int = None

    def __post_init__(self):
        if self.total_time is None:
            self.total_time = len(self.driving_data)

def run_simulation(system: Any, controller: Any, config: SimConfiguration, verbose: int=0):
    '''Simulation loop that works with any controller that has a .compute_control() method'''
    time_steps = np.arange(0, config.total_time, config.dt)
    results_list = []

    print(f"- Simulation {type(controller).__name__}")
    start = time.time()
    for i, t in enumerate(time_steps):
        idx = min(i, len(config.driving_data)-1)
        current_p_driv = config.driving_data[idx]
        current_velocity = config.velocity_data[idx]

        disturbances = np.array([current_p_driv, config.T_amb])

        controls = controller.compute_control(system.state, disturbances, current_velocity)
        next_state, diagnostics = system.step(controls, disturbances, config.dt)

        record = {
            'time': t,
            'T_batt': system.state[0],
            'T_clnt': system.state[1],
            'soc': system.state[2],
            'w_comp': controls[0],
            'w_pump': controls[1],
            'P_driv': current_p_driv,
            # Flatten diagnostics into the record
            **diagnostics 
        }
        results_list.append(record)
        if verbose != 0:
            np.set_printoptions(precision=2)
            print_log = verbose * 100
            if (i % print_log) == 0:
                print(f"    > Step {i}, P_cool = {diagnostics['P_cooling'] / 1000:.5f} kJ, Tbatt = {system.state[0]:.2f} °C, w = {controls} rpm --- DEBUG: P_driv = {current_p_driv / 1000 :.2f} kJ")

    print(f"    Simulation finished")
    sim_time = time.time()-start
    print(
        f"[{type(controller).__name__}] "
        f"Total simulation time: {sim_time:.3f} s | "
        f"Average time per step (dt = {config.dt} s): {sim_time/len(config.driving_data):.6f} s"
    )


    results_df = pd.DataFrame(results_list)
    return results_df

