# Battery Thermal Management (BTM)
This folder contains multiple `.py` files that I used for a Stochastic MPC (SMPC) controller. In SMPC the main idea is to treat external disturbances as a random variable (RV) which is bounded to a set $M$, with a computed density distribution.

## Key Features:
- Stochastic Model Predictive Control formulation
- Robust constraint handling under uncertainty
- Battery thermal dynamics modeling
- Disturbance rejection for thermal management

## Files Included:
- `sys_dynamics.py` - System dynamics which connect all other needed computations for the system
- `battery_models.py` - Battery thermo-electrical models
- `entropy.py` - (reversible) Battery entropic heat data generation yields a `.npz` file
- `driving_energy.py` - Energy used for a cycling driving data (i.e., UDDS), yields a `.npz` file