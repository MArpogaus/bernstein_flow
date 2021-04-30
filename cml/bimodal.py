#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : bimodal.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-03-22 16:42:31 (Marcel Arpogaus)
# changed : 2021-04-30 18:53:50 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################

from functools import partial
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from bernstein_flow.bijectors import BernsteinBijector
from bernstein_flow.distributions import BernsteinFlow
from bernstein_flow.util.visualization import vizualize_flow_from_z_domain

# Ensure Reproducibility
np.random.seed(2)
tf.random.set_seed(2)
print("TFP Version", tfp.__version__)
print("TF  Version", tf.__version__)


def print_param(b, indent=0, prefix=""):
    s = " " * indent + prefix
    if not isinstance(b, tfb.Bijector):
        s += f"{b.name}:\n"
        s += print_param(b.bijector, indent + 4, prefix)
    elif isinstance(b, tfb.Invert):
        s += f"{b.name}:\n"
        s += print_param(b.bijector, indent + 4, prefix)
    elif isinstance(b, tfb.Chain):
        s += f"{b.name}:\n"
        s += "".join(
            map(partial(print_param, indent=indent + 4, prefix=prefix), b.bijectors)
        )
    elif isinstance(b, tfb.Scale):
        s += f"{b.name}: {b.scale}\n"
    elif isinstance(b, tfb.Shift):
        s += f"{b.name}: {b.shift}\n"
    elif isinstance(b, BernsteinBijector):
        s += f"{b.name}: {b.theta}\n"
    elif isinstance(b, (tfb.Sigmoid, tfb.SoftClip)):
        s += f"{b.name}: {b.high}, {b.low}\n"
    else:
        s += f"{b.name}: No Params\n"
    return s


def gen_data(t):
    t1 = t
    y1 = 1.0 * t1
    y1 += np.random.normal(0, 0.05 * np.abs(t1))

    t2 = t
    y2 = -0.2 * t2
    y2 += np.random.normal(0, 0.2 * np.abs(t2))

    t = np.concatenate([t1, t2])
    y = np.concatenate([y1, y2])

    return t[..., np.newaxis], y[..., np.newaxis]


def gen_test_data(n=5, observations=100):
    t = np.linspace(0, 1, n, dtype=np.float32)
    t = np.repeat([t], observations)

    return gen_data(t)


def gen_train_data(n=100):
    t = np.random.uniform(0, 1, n // 2).astype(np.float32)

    return gen_data(t)


def gen_model(bernstein_order=9, **kwds):
    tf.random.set_seed(1)
    output_shape = 2 + bernstein_order
    if kwds.get("second_affine_trafo", True):
        output_shape += 1

    flow_parameter_model = Sequential(
        [
            Input(1),
            Dense(16, activation="relu"),
            Dense(16, activation="relu"),
            Dense(output_shape),
        ]
    )

    def bf(y_pred):
        return BernsteinFlow.from_pvector(
            y_pred, allow_values_outside_support=True, **kwds
        )

    def my_loss_fn(y_true, y_pred):
        return -tfd.Independent(bf(y_pred)).log_prob(tf.squeeze(y_true))

    flow_parameter_model.compile(
        optimizer="adam",
        loss=my_loss_fn
        # run_eagerly=True
    )
    return flow_parameter_model, bf


def fit_model(**kwds):
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=3
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=9, restore_best_weights=True
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ] + kwds.pop("callbacks", [])
    model, bf = gen_model(**kwds)
    hist = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=300,
        shuffle=True,
        batch_size=32,
        callbacks=callbacks,
    )
    return model, bf, hist


def plot_dists(model, bf):
    flow = bf(model(test_x[..., None]))
    yy = np.linspace(-1, 1.5, 1000, dtype=np.float32)[..., None]
    ps = flow.prob(yy)

    fig, ax = plt.subplots(len(test_x), figsize=(16, len(test_x) * 4))
    for i, x in enumerate(test_x):
        ax[i].set_title(f"x={x}")
        sampl = test_y[(test_t.flatten() == x)].flatten()
        ax[i].scatter(sampl, [0] * len(sampl), marker="|")
        # ax[i].hist(sampl, bins=30, density=True)
        ax[i].plot(yy, ps[:, i])

    return fig


def plot_chained_bijectors(flow):
    chained_bijectors = flow.bijector.bijector.bijectors
    base_dist = flow.distribution
    cols = len(chained_bijectors) + 1
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))

    n = 200

    z_samples = np.linspace(-3, 3, n).astype(np.float32)
    log_probs = base_dist.log_prob(z_samples)

    ax[0].plot(z_samples, np.exp(log_probs))

    zz = z_samples[..., None]
    ildj = 0.0
    for i, (a, b) in enumerate(zip(ax[1:], chained_bijectors)):
        # we need to use the inverse here since we are going from z->y!
        z = b.inverse(zz)
        ildj += b.forward_log_det_jacobian(z, 1)
        # print(z.shape, zz.shape, ildj.shape)
        a.scatter(z, np.exp(log_probs + ildj))
        a.set_title(b.name.replace("_", " "))
        a.set_xlabel(f"$z_{i}$")
        a.set_ylabel(f"$p(z_{i+1})$")
        zz = z
    fig.tight_layout()
    return fig


def plot_x_trafo(flow, xmin=-1, xmax=1, n=20, size=4):
    x = np.linspace(xmin, xmax, n, dtype=np.float32)
    num_bij = len(flow.bijector.bijector.bijectors)
    fig, ax = plt.subplots(2, num_bij, figsize=(size * num_bij, size * 2))
    for i, b in enumerate(reversed(flow.bijector.bijector.bijectors)):
        y = b.forward(x)
        ax[0, i].scatter(x, y, alpha=0.2)
        ax[0, i].set_title(
            b.name.replace("_", " ").title().replace("Bernsteinflow", "")
        )
        x = y
    # y = np.linspace(min(y),max(y), 400, dtype=np.float32)
    for i, b in enumerate(flow.bijector.bijector.bijectors):
        x = b.inverse(tf.identity(y))
        ax[1, num_bij - i - 1].scatter(x, y, alpha=0.2)
        ax[1, num_bij - i - 1].set_title(
            b.name.replace("_", " ").title().replace("Bernsteinflow", "")
        )
        y = x

    return fig


def plot_value_and_gradient(func, y):
    [funval, grads] = tfp.math.value_and_gradient(func, y)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].scatter(y, funval, s=1, label="funval")
    ax[0].legend()
    ax[1].scatter(y, grads, s=1, label="grad")
    ax[1].legend()
    return fig


# Data


n = 4000
train_x, train_y = gen_train_data(n=n)
val_x, val_y = gen_train_data(n=n // 10)
train_x.shape, train_y.shape, val_x.shape, val_y.shape
test_t, test_y = gen_test_data(5, 200)
test_x = np.unique(test_t)

fig = plt.figure(figsize=(16, 8))
plt.scatter(train_x, train_y, alpha=0.5, label="train")
plt.scatter(test_t, test_y, alpha=0.5, label="test")
plt.scatter(val_x, val_y, alpha=0.5, label="test")

plt.legend()
fig.savefig("bm_data.png")

# Fit Model

model, bf, hist = fit_model(bernstein_order=20, scale_base_distribution=True)


# Results
result_path = "metrics/"
figure, axes = plt.subplots(2, figsize=(16, 8))
hist_df = pd.DataFrame(hist.history)
hist_df.to_csv(result_path + "bm_hist.csv")

fig = hist_df[["loss", "val_loss"]].plot(ax=axes[0]).get_figure()
hist_df[["lr"]].plot(ax=axes[1])
fig.savefig(result_path + "bm_hist.png")

with open(result_path + "bm_metrics.txt", "w") as metrics:
    metrics.write("loss: " + str(hist_df.loss.min()) + "\n")
    metrics.write("val loss: " + str(hist_df.val_loss.min()) + "\n")

fig = plot_dists(model, bf)
fig.savefig(result_path + "bm_dists.png")

flow = bf(model(tf.ones((1, 1))))


with open(result_path + "bm_pvector.txt", "w") as pvector:
    pvector.write(print_param(flow))

fig = vizualize_flow_from_z_domain(flow)
fig.savefig(result_path + "bm_flow.png")


# Bijector


fig = plot_x_trafo(flow, xmin=-2, xmax=2, n=25)
fig.savefig(result_path + "bm_bijectors.png")

fig = plot_chained_bijectors(flow)
fig.savefig(result_path + "bm_trafo.png")


y = np.linspace(-3, 3, 1000, dtype=np.float32)
fig = plot_value_and_gradient(flow.bijector.inverse, y.copy())
fig.savefig(result_path + "bm_bijector.png")


def fun(y):
    return flow.bijector.inverse_log_det_jacobian(y, 0)


fig = plot_value_and_gradient(fun, y.copy())
fig.savefig(result_path + "bm_ildj.png")
