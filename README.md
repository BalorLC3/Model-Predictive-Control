## Battery Thermal Management System (BTMS)

This repository contains a collection of Python modules developed to study and benchmark advanced control strategies for **battery thermal management in electric vehicles** using a **high-fidelity lithium-ion battery model**.

A primary focus of this work is **Stochastic Model Predictive Control (SMPC)**, where exogenous disturbances are modeled as **random variables bounded within a compact set** ( M ), with known or estimated probability density functions. This formulation enables explicit handling of uncertainty while maintaining constraint satisfaction with probabilistic guarantees.

In contrast, **reinforcement learning (RL)** approaches do not rely on explicit disturbance models. Instead, an agent interacts directly with the environment, progressively learning the system dynamics by balancing exploration and exploitation in order to maximize a cumulative reward.

This project provides a **high-fidelity simulation environment** suitable for fair and consistent evaluation of both **model-based** and **data-driven** controllers under identical thermal and operational conditions.

---

## Key Features

All control strategies are implemented and evaluated on the same high-fidelity BTMS model to ensure meaningful performance comparisons. The implemented methods include:

* **Stochastic Model Predictive Control (SMPC)** with explicit uncertainty modeling
* **Deterministic Dynamic Programming (DP)**
* **Backward Dynamic Programming** (non-causal optimal benchmark)
* **Reinforcement Learning**, using policy-gradient methods

The framework is designed to highlight trade-offs in **performance, robustness, optimality, and implementability** across different control paradigms.

---

## Repository Structure

```text
Model-Predictive-Control_Battery-Management-System
├── classical-mpc              # Baseline MPC implementations (contextual reference)
├── dynamic-prog-and-rl        # Implemented in JAX
│   ├── controllers            # Reinforcement Learning (Soft Actor-Critic)
├── stochastic-mpc             # Implemented in CasADi (symbolic formulation)
│   ├── controllers            # SMPC and thermostat-based controllers
├── utils-dynamics             # Battery and thermal dynamics utilities
└── results_btm                # Simulation results and visualizations
```

---

## Results

The following figures compare temperature regulation performance across different control strategies:

| Thermostat                                                                                                                    | Dynamic Programming                                                                                          | SMPC                                                                                                                 | SAC                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| <img src="xresults_btm/thermostat_controller.png" width="200" alt="Thermostat Controller"><br>*Baseline heuristic controller* | <img src="xresults_btm/dp_controller.png" width="200" alt="DP Controller"><br>*Non-causal optimal benchmark* | <img src="xresults_btm/smpc_controller.png" width="200" alt="SMPC Controller"><br>*Stochastic MPC under uncertainty* | <img src="xresults_btm/sac_h0_controller.png" width="200" alt="SAC Controller"><br>*Soft Actor-Critic (model-free RL)* |

The **dynamic programming solution** serves as a non-implementable lower bound on achievable cost, while **SMPC** demonstrates robust constraint handling under uncertainty. The **SAC controller** highlights the potential and limitations of model-free learning when applied to safety-critical thermal systems.

---

