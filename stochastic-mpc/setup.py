from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SimConfiguration:
    """Holds the scenario parameters to ensure all controllers run on identical conditions"""
    driving_data: np.array
    T_amb: float
    dt: float = 1.0
    total_time: int = None

    def __post_init__(self):
        if self.total_time is None:
            self.total_time = len(self.driving_data)

def run_simulation(system, controller, config: SimConfiguration):
    '''Simulation loop that works with any controller that has a .compute_control() method'''
    time_steps = np.arange(0, config.total_time, config.dt)
    results_list = []

    print(f"Simulation {type(controller).__name__}")
    for i, t in enumerate(time_steps):
        idx = min(i, len(config.driving_data)-1)
        current_p_driv = config.driving_data[idx]

        disturbances = np.array([current_p_driv, config.T_amb])

        controls = controller.compute_control(system.state, disturbances)
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

    print(f"Simulation finished")

    results_df = pd.DataFrame(results_list)
    return results_df
