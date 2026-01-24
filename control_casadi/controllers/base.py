from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    @abstractmethod
    def compute_control(self, state: np.ndarray, 
                       disturbance: np.ndarray = None,
                       **kwargs) -> np.ndarray:
        """
        Compute control action for current state.
        
        Args:
            state: Current system state (numpy array)
            disturbance: Current disturbance (optional)
            **kwargs: Additional parameters (reference, time, etc.)
            
        Returns:
            control: Control action (numpy array)
        """
        raise NotImplementedError