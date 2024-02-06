import numpy as np
from bernstein_flow.math.bernstein import derive_bpoly, gen_bernstein_polynomial

np.random.seed(1)


def test_approximation():
    M = 500
    x = np.linspace(0, 1, M)
    y = np.sin(x)

    theta = y

    bpoly, _ = gen_bernstein_polynomial(theta)
    dbpoly, _ = derive_bpoly(theta)

    eps = 1e-5
    x_test = np.random.uniform(0 + eps, 1 - eps, 1000)
    y_true = np.sin(x_test)
    y_approx = bpoly(x_test)
    assert np.allclose(y_true, y_approx, 1e-4, 1e-3), "Approximation failed"

    dy_true = np.cos(x_test)
    dy_approx = dbpoly(x_test)
    assert np.allclose(dy_true, dy_approx, 1e-4, 1e-3), "Wrong Derivative"
