# Battery Thermal Management (BTM)
This folder contains multiple `.py` files that I used for a Stochastic MPC (SMPC) controller. In SMPC the main idea is to treat external disturbances as a random variable (RV) which is bounded to a set $M$, with a computed density distribution.

With JAX and Casadi implementation.

## Key Features:
All control strategies were made for a high fidelity BTMS model, these strategies include the following:
- Stochastic Model Predictive Control formulation
- Deterministic Dynamic Programming
- Backward Dynamic Programming
- Reinforcement Learning

## Context
'''
Model-Preditive-Control_Battery-Management-System \\
├── classical-mpc \\
├── dynamic-prog-and-rl \\
├── stochastic-mpc \\
├── utils-dynamics \\
└── results_btm
'''
## Results
