import numpy as np


def grad_f(x, A, b):
    return A.T @ (A @ x - b)

def proj_ball(y, radius=1.0):
    norm_y = np.linalg.norm(y)
    if norm_y <= radius:
        return y
    return radius * y / norm_y

    
def projected_gradient_normalized(
    grad_f,
    proj_C,
    x0,
    lambdas,
    max_iter=1000,
    tol=1e-6
):
    """
    Projected Gradient Descent with Normalized Stepsize

    Parameters
    ----------
    grad_f : function
        Gradient of f, grad_f(x)
    proj_C : function
        Projection onto C, proj_C(y)
    x0 : np.ndarray
        Initial point in C
    lambdas : iterable
        Sequence lambda_n with sum lambda_n = inf, sum lambda_n^2 < inf
    max_iter : int
        Maximum number of iterations
    tol : float
        Stopping tolerance

    Returns
    -------
    x : np.ndarray
        Approximate solution
    history : list
        History of iterates
    """

    x = x0.copy()
    history = [x.copy()]

    for n in range(min(max_iter, len(lambdas))):
        g = grad_f(x)
        norm_g = np.linalg.norm(g)

        alpha = lambdas[n] / max(1.0, norm_g)

        x_next = proj_C(x - alpha * g)

        if np.linalg.norm(x_next - x) < tol:
            break

        x = x_next
        history.append(x.copy())

    return x, history
np.random.seed(0)
n, m = 5, 8
A = np.random.randn(m, n)
b = np.random.randn(m)

# Gradient wrapper
def grad(x):
    return grad_f(x, A, b)

# Initial point
x0 = np.zeros(n)

# Lambda_n = 1 / n
N = 5000
lambdas = np.array([1.0 / (k + 1) for k in range(N)])

# Run algorithm
x_star, history = projected_gradient_normalized(
    grad_f=grad,
    proj_C=lambda y: proj_ball(y, radius=1.0),
    x0=x0,
    lambdas=lambdas,
    max_iter=2000
)

print("Solution:", x_star)
print("Norm:", np.linalg.norm(x_star))

values = [0.5 * np.linalg.norm(A @ x - b)**2 for x in history]

import matplotlib.pyplot as plt
plt.plot(values)
plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.title("Convergence of Normalized PGD")
plt.grid(True)
plt.show()