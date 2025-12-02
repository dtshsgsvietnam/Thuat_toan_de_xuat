import numpy as np


def proj_C(x, R=10):
    return np.clip(x, -R, R)


def SGDA(
    x0,
    f_xi,
    grad_f_xi,
    proj_C,
    lam0=1.0,      # λ_0
    sigma=0.5,     # σ ∈ (0,1)
    kappa=0.5,     # κ ∈ (0,1)
    max_iter=1000
):
    xk = x0.copy()
    lam_k = lam0

    for k in range(max_iter):

        # 1. Sample ξ_k
        xi_k = np.random.randint(0, 1000000)  # random seed or data index

        # 2. Compute gradient & candidate update
        grad = grad_f_xi(xk, xi_k)
        x_new = proj_C(xk - lam_k * grad)

        # 3. Adaptive rule cho λ_{k+1}
        if f_xi(x_new, xi_k) <= f_xi(xk, xi_k) - sigma * np.dot(grad, xk - x_new):
            lam_next = lam_k
        else:
            lam_next = kappa * lam_k

        # 4. Check stopping
        if np.allclose(xk, x_new):
            print("Dừng tại k =", k)
            break

        # Cập nhật cho vòng sau
        xk = x_new
        lam_k = lam_next

    return xk
