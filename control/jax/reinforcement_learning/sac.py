import flax.linen as nn


class SBXActor(nn.Module):
    n_actions: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x
        