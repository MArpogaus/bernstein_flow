# %% imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.bijectors import BernsteinPolynomial
from matplotlib import pyplot as plt

# %% Globals
M = 25
batch_size = 10
tf.random.set_seed(1)
# creates a random parameter vector
thetas = tf.random.uniform((1, M), -3, 2, dtype=tf.float32)

# %% Test no support
tcf = get_thetas_constrain_fn(bounds=(-3, 3), smooth_bounds=True)
bpoly = BernsteinPolynomial(thetas=tcf(thetas), extrapolation=True)

x = tf.cast(tf.linspace(-0.5, 1.5, 2000), tf.float32)
[y, y_grad] = tfp.math.value_and_gradient(bpoly, x)

xx = bpoly.inverse(tf.identity(y))

fldj = bpoly.forward_log_det_jacobian(x, 0).numpy()
J = np.log(np.abs(np.gradient(y, np.diff(x).mean())))

ildj = bpoly.inverse_log_det_jacobian(y, 0).numpy()
iJ = np.log(abs(np.gradient(x)))

f"{x.shape=}, {xx.shape=}, {y.shape=}, {y_grad.shape=} {fldj.shape=}, {J.shape=}, {ildj.shape=}, {iJ.shape=}"

# %% Plot
fig, axs = plt.subplots(2, sharex=True)
fig.suptitle("Bernstein polynomial and inverse with extrapolation")
axs[0].plot(x, y, label="Bernstein polynomial")
# axs[0].scatter(
#     tf.linspace(-10, 10, bpoly.order + 1),
#     bpoly.theta.numpy().flatten(),
#     label="Bernstein coefficients",
# )
axs[0].plot(xx, y, ":", label="inverse")
axs[0].legend()
axs[1].plot(x, fldj, label="fldj")
axs[1].plot(x, np.log(np.abs(y_grad)), ":", label="y_grad")
# axs[1].plot(x, -ildj, label="-ifldj")
# axs[1].scatter(
#     tf.linspace(-10, 10, bpoly.order),
#     bpoly.dtheta.numpy().flatten(),
#     label="dtheta",
# )
# axs[1].plot(x, J, ":", label="ladj (numpy)")
axs[1].legend()
fig.tight_layout()
fig.savefig("extra_with_invers.png")

# %% inverse ldaj
yy = tf.cast(tf.linspace(-3, 3, 2000), tf.float32)
xx = bpoly.inverse(yy)

ildj = bpoly.inverse_log_det_jacobian(yy, 0).numpy()
iJ = np.log(abs(np.gradient(xx, np.diff(yy).mean())))

# %% Plot

fig, axs = plt.subplots(2, sharex=True)
fig.suptitle("Bernstein polynomial and inverse with extrapolation")
axs[0].plot(yy, xx, label="inverse")
axs[0].legend()
axs[1].plot(yy, ildj, label="ildj")
axs[1].plot(yy, iJ, ":", label="iladj (numpy)")
axs[1].legend()
fig.tight_layout()
fig.savefig("extra_with_invers.png")

# %% Test support
tcf = get_thetas_constrain_fn(bounds=(-3, 3), smooth_bounds=True)
bpoly = BernsteinPolynomial(
    thetas=tcf(thetas), extrapolation=True, analytic_jacobian=True, domain=(-5, 15)
)

x = tf.cast(tf.linspace(-5.5, 15.5, 2000), tf.float32)
[y, y_grad] = tfp.math.value_and_gradient(bpoly, x)

xx = bpoly.inverse(tf.identity(y))

fldj = bpoly.forward_log_det_jacobian(x, 0).numpy()
J = np.log(np.abs(np.gradient(y, np.diff(x).mean())))

ildj = bpoly.inverse_log_det_jacobian(y, 0).numpy()
iJ = np.log(abs(np.gradient(x)))

f"{x.shape=}, {xx.shape=}, {y.shape=}, {y_grad.shape=} {fldj.shape=}, {J.shape=}, {ildj.shape=}, {iJ.shape=}"


# %% Plot
fig, axs = plt.subplots(2, sharex=True)
fig.suptitle("Bernstein polynomial and inverse with extrapolation")
axs[0].plot(x, y, label="Bernstein polynomial")
# axs[0].scatter(
#     tf.linspace(-10, 10, bpoly.order + 1),
#     bpoly.theta.numpy().flatten(),
#     label="Bernstein coefficients",
# )
axs[0].plot(xx, y, ":", label="inverse")
axs[0].legend()
axs[1].plot(x, fldj, label="fldj")
axs[1].plot(x, np.log(np.abs(y_grad)), ":", label="y_grad")
# axs[1].plot(x, -ildj, label="-ifldj")
# axs[1].scatter(
#     tf.linspace(-10, 10, bpoly.order),
#     bpoly.dtheta.numpy().flatten(),
#     label="dtheta",
# )
# axs[1].plot(x, J, ":", label="ladj (numpy)")
axs[1].legend()
fig.tight_layout()
fig.savefig("extra_with_invers.png")
