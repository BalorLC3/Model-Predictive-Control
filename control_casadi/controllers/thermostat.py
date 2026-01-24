import casadi as ca
import numpy as np
from controllers.base import BaseController

class Thermostat(BaseController):
    '''Rule based controller; threshold based'''
    def __init__(self):
        self.cooling_active = False

    def compute_control(self, state, disturbance, velocity=0.0):
        T_batt, _, _ = state
        T_upper, T_lower = 33.5, 32.0
        
        if T_batt > T_upper:
            self.cooling_active = True
        elif T_batt < T_lower:
            self.cooling_active = False
        w_pump = 4000.0 if self.cooling_active else 0.0
        w_comp = 3000.0 if self.cooling_active else 0.0

        return w_comp, w_pump