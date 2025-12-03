import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy
from abc import ABC, abstractmethod

class BaseController(ABC):
    @abstractmethod
    def compute_control(self, state, disturbance):
        pass

import numpy as np

class Thermostat(BaseController):
    '''Rule based controller; threshold based'''
    def __init__(self):
        self.cooling_active = False

    def compute_control(self, state, disturbance):
        T_batt, _, _ = state
        T_upper, T_lower = [34.0, 32.5]
        
        self.cooling_active = np.where(
            T_batt > T_upper, 
            True, 
            np.where(T_batt < T_lower, False, self.cooling_active)
        )
        
        w_pump = np.where(self.cooling_active, 2000.0, 0.0)
        w_comp = np.where(self.cooling_active, 3000.0, 0.0)
        
        return w_comp, w_pump
    
class Constraints():
    def __init__(self):
        self.p = 0
        
class SMPC():
    pass