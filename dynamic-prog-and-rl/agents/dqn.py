import jax, jax.numpy as jnp
from functools import partial   
import flax, flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, Callable, Any, Tuple   
import optax
from system.jax_ode_solver import rk4_step

def epsilon_decay(
        init_eps: float, # Initial epsilon
        min_eps: float,  # Minimal epsilon for exploration
        duration: int, 
        t: int # Current time
    ):
    '''Linear Scheduling decay for exploration'''
    slope = (min_eps - init_eps) / duration
    return max(slope * t + init_eps, min_eps)

class QNetwork(nn.Module):
    layer_sizes: Sequence[int]
    n_actions: int
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
        q_values = nn.Dense(features=self.n_actions)(x)
        return q_values 
    
class TrainState(TrainState):
    target_params: flax.core.FrozenDict




