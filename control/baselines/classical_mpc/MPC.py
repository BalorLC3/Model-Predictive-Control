from lq_gain import lq_gain
import casadi as ca
import numpy as np

def MPC_casadi(F, G, A, B, Q, R, VT, x0, N=6):
    K, P = lq_gain(A, B, Q, R)
    nx = A.shape[0]
    nu = B.shape[1]
    
    opti = ca.Opti()
    c = opti.variable(nu, N)  
    x = opti.variable(nx, N+1)
    # Precompute matrices
    FGK = F + G @ K               
    AK = A + B @ K                

    cost = 0
    opti.subject_to(x[:, 0] == x0)

    for i in range(N):
        opti.subject_to(x[:, i+1] == AK @ x[:, i] + B @ c[:, i])

        u_i = K @ x[:, i] + c[:, i]  
        cost += x[:, i].T @ Q @ x[:, i] + u_i.T @ R @ u_i

        opti.subject_to(FGK @ x[:, i] + G @ c[:, i] <= np.ones(F.shape[0]))

    cost += x[:, N].T @ P @ x[:, N]
    opti.subject_to(VT @ x[:, N] <= np.ones(VT.shape[0]))

    opti.minimize(cost)

    p_opts = {"print_time": False, "verbose": False}
    s_opts = {"max_iter": 1000, "tol": 1e-6, "print_level": 0}
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        c_opt = sol.value(c)
        x_opt = sol.value(x)
        u0 =  K @ x0 + c_opt[0]

        J = sol.value(cost)
        return J, float(u0), c_opt, x_opt
    
    except Exception as e:
        print(f"MPC failed: {e}")
        try:
            c_opt = opti.debug.value(c)
            x_opt = opti.debug.value(x)
            if c_opt is not None and x_opt is not None:
                u0 = (K @ x0 + c_opt[0]).flatten()
                J = opti.debug.value(cost)
                print("Returning suboptimal solution")
                return J, u0, c_opt.T, x_opt
        except:
            pass
        return None, None, None, None