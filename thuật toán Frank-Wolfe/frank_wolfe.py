import numpy as np

def frank_wolfe(
        f, grad_f, linear_minimize,
        x0, max_iter=1000):

    x = x0.copy()
    history = [x.copy()]

    for k in range(max_iter):
        g = grad_f(x)

        # Step 1: linear minimization oracle
        s = linear_minimize(g)

        # Step 2: cỡ bước
        gamma = 2.0 / (k + 2)

        # Step 3: cập nhật x
        x = (1 - gamma) * x + gamma * s

        history.append(x.copy())

    return x, history


# ====== DEMO =======
def f(x):
    return (x[0]-1)**2 + (x[1]+2)**2

def grad_f(x):
    return np.array([2*(x[0]-1), 2*(x[1]+2)])

# Linear oracle trên hộp [-1,2]×[-3,1]
def linear_minimize(g):
    low  = np.array([-1, -3])
    high = np.array([ 2,  1])
    return np.where(g > 0, low, high)

x0 = np.array([2.0, 2.0])
x_star, hist = frank_wolfe(f, grad_f, linear_minimize, x0)
print("Kết quả:", x_star)
