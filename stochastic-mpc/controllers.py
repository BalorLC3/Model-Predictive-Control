import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy
from abc import ABC, abstractmethod

class BaseController(ABC):
    @abstractmethod
    def compute_control(self, state, disturbance):
        pass

class Thermostat(BaseController):
    '''Rule based controller; threshold based'''
    def __init__(self):
        self.cooling_active = False

    def compute_control(self, state, disturbance):
        T_batt, T_clnt, soc = state
        # Thresholds
        T_upper, T_lower = [34.0, 32.5]

        if T_batt > T_upper: # Upper temperature threshold [celsius]
            self.cooling_active = True
        elif T_batt < T_lower: # Lower temperature threshold 
            self.cooling_active = False

        if self.cooling_active:
            w_pump = 2000 # [rpm]
            w_comp = 3000 
        else:
            w_pump = 0 # [rpm]
            w_comp = 0 
        return np.array([w_comp, w_pump])


class Constraints():
    def __init__(self):
        self.rho_
        
class SMPC():
    pass