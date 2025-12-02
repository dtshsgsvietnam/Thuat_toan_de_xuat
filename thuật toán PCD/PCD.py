import numpy as np

# Projection vào C = [0,1]^n
def proj_C(x):
    return np.clip(x, 0.0, 1.0)

# PCD thuật toán
def projected_coordinate_descent(f_grad, x0, alpha=0.1, steps=200, random=False):
    x = x0.copy()
    n = len(x)

    for k in range(steps):
        # Chọn 1 coordinate
        if random:
            i = np.random.randint(n)
        else:
            i = k % n

        grad = f_grad(x)            # toàn bộ gradient
        e = np.zeros(n)             # vector đơn vị
        e[i] = 1.0

        x = x - alpha * grad[i] * e  # cập nhật 1 chiều
        x = proj_C(x)                # chiếu vào C

    return x