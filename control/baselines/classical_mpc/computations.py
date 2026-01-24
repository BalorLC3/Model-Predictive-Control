from scipy.linalg import solve_discrete_are
import cvxpy as cp
import numpy as np
# ---- K-LQ Gain matrix and P of riccati equation ----

def compute_nu(F, G, K, A, B, n_max=10):
    FGK = F + G @ K
    AK = A + B @ K
    nC = FGK.shape[0]
    dim_x = A.shape[0]
    x = cp.Variable(dim_x)

    for n in range(n_max):
        # create constraints for i = 0 to n
        constraints = [(FGK @ np.linalg.matrix_power(AK, i)) @ x <= np.ones(nC) for i in range(n + 1)]

        Phi_np1 = np.linalg.matrix_power(AK, n + 1)
        expr = (FGK @ Phi_np1) @ x

        # for each row j, maximize row_j * Phi^(n+1) x
        max_violations = []
        for j in range(FGK.shape[0]):
            prob = cp.Problem(cp.Maximize(expr[j]), constraints)
            prob.solve(cp.SCS)
            if prob.status != "optimal":
                return n  # infeasible earlier
            max_violations.append(prob.value)  

        # If all rows <= 1, then invariant reached
        if all(v <= 1.0 + 1e-6 for v in max_violations):
            return n

    return n_max

def compute_VT(F, G, K, A, B):
    ''' v and VT for stability constraint'''
    nu = compute_nu(F, G, K, A, B, 10)  
    FGK = F + G @ K
    AK = A + B @ K
    VT_blocks = []
    for i in range(nu + 1):
        VT_blocks.append(FGK @ np.linalg.matrix_power(AK, i))
    return np.vstack(VT_blocks)