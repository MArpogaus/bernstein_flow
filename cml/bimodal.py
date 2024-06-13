#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION #####################################################
# file    : bimodal.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-03-22 16:42:31 (Marcel Arpogaus)
# changed : 2024-02-29 13:36:07 (Marcel Arpogaus)
# DESCRIPTION ############################################################
# ...
# LICENSE ################################################################
# ...
##########################################################################
# %% Imports
import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from bernstein_flow.distributions import BernsteinFlow
from bernstein_flow.util.visualization import (
    plot_chained_bijectors,
    plot_flow,
    plot_value_and_gradient,
    plot_x_trafo,
)
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow_probability import bijectors as tfb

try:
    import mlflow

    USE_MLFLOW = True
except ImportError:
    USE_MLFLOW = False

# %% globals
metrics_path = "metrics/bimodal"
artifacts_path = "artifacts/bimodal"


# %% functions
def print_param(b, indent=0, prefix=""):
    s = " " * indent + prefix
    if not isinstance(b, tfb.Bijector):
        s += f"{b.name}:\n"
        s += print_param(b.bijector, indent + 4, prefix)
    elif isinstance(b, tfb.invert._Invert):
        s += f"{b.name}:\n"
        s += print_param(b.bijector, indent + 4, prefix)
    elif isinstance(b, tfb.chain._Chain):
        s += f"{b.name}:\n"
        s += "".join(
            map(partial(print_param, indent=indent + 4, prefix=prefix), b.bijectors)
        )
    elif isinstance(b, tfb.Scale):
        s += f"{b.name}: {b.scale}\n"
    elif isinstance(b, tfb.Shift):
        s += f"{b.name}: {b.shift}\n"
    elif hasattr(b, "thetas"):
        s += f"{b.name}: {b.thetas}\n"
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


def scale_data(y, min_y, max_y):
    return (y - min_y) / (max_y - min_y)


def gen_test_data(n=5, observations=100):
    t = np.linspace(0, 1, n, dtype=np.float32)
    t = np.repeat([t], observations)

    return gen_data(t)


def gen_train_data(n=100):
    t = np.random.uniform(0, 1, n // 2).astype(np.float32)

    return gen_data(t)


def gen_model(output_shape=9, **kwds):
    flow_parameter_model = Sequential(
        [
            Input(1),
            Dense(16, activation="relu"),
            Dense(16, activation="relu"),
            Dense(output_shape, activation="linear"),
        ]
    )

    def bf(y_pred):
        return BernsteinFlow.new(y_pred, **kwds)

    def my_loss_fn(y_true, y_pred):
        dist = bf(y_pred)
        return -dist.log_prob(tf.squeeze(y_true))

    flow_parameter_model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=my_loss_fn,
        jit_compile=True,
        # run_eagerly=True,
    )
    return flow_parameter_model, bf


def fit_model(
    model,
    bf,
    train_x,
    train_y,
    val_x,
    val_y,
    batch_size,
    epochs,
    lr_patience=15,
    **kwds,
):
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=lr_patience
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3 * lr_patience + 10, restore_best_weights=True
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ] + kwds.pop("callbacks", [])
    hist = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks,
        **kwds,
    )
    return hist


def plot_dists(model, bf, test_x, test_t, test_y):
    flow = bf(model(test_x[..., None]))
    yy = np.linspace(-1, 1.5, 1000, dtype=np.float32)[..., None]
    ps = flow.prob(yy)

    fig, ax = plt.subplots(
        len(test_x), figsize=(10, len(test_x) * 3), constrained_layout=True
    )
    fig.suptitle("Learned Distributions", fontsize=16)

    for i, x in enumerate(test_x):
        ax[i].set_title(f"x={x}")
        sampl = test_y[(test_t.flatten() == x)].flatten()
        ax[i].scatter(sampl, [0] * len(sampl), marker="|")
        ax[i].plot(yy, ps[:, i], label="flow")
        ax[i].set_xlabel("y")
        ax[i].set_ylabel(f"p(y|x={x})")
        ax[i].legend()
    return fig


def prepare_data(n=100, scale_data_to_domain=False):
    # Data
    train_x, train_y = gen_train_data(n=n)
    val_x, val_y = gen_train_data(n=n // 10)
    test_x, test_y = gen_test_data(5, 200)

    if scale_data_to_domain:
        min_y, max_y = train_y.min(), train_y.max()
        train_y = scale_data(train_y, min_y, max_y)
        val_y = scale_data(val_y, min_y, max_y)
        test_y = scale_data(test_y, min_y, max_y)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def results(
    model,
    bf,
    hist,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    test_t,
    metrics_path,
    artifacts_path,
):
    fig = plt.figure(figsize=(16, 8))
    plt.scatter(train_x, train_y, alpha=0.5, label="train")
    plt.scatter(test_t, test_y, alpha=0.5, label="test")
    plt.scatter(val_x, val_y, alpha=0.5, label="validate")

    plt.legend()
    fig.savefig(os.path.join(artifacts_path, "bm_data.png"))

    fig, axes = plt.subplots(2, figsize=(16, 8))
    hist_df = pd.DataFrame(hist.history)

    fig = hist_df[["loss", "val_loss"]].plot(ax=axes[0]).get_figure()
    hist_df[["lr"]].plot(ax=axes[1])
    fig.savefig(os.path.join(artifacts_path, "bm_hist.png"))

    with open(os.path.join(metrics_path, "bm_metrics.yaml"), "w") as metrics:
        min_loss = hist_df[["loss", "val_loss"]].min().to_dict()
        yaml.dump(min_loss, metrics)

    fig = plot_dists(model, bf, test_x, test_t, test_y)
    fig.savefig(os.path.join(artifacts_path, "bm_dists.png"))

    flow = bf(model(tf.ones((1, 1))))

    with open(os.path.join(artifacts_path, "bm_pvector.txt"), "w") as pvector:
        pvector.write(print_param(flow))

    fig = plot_flow(flow)
    fig.savefig(os.path.join(artifacts_path, "bm_flow.png"))

    # Bijector
    fig = plot_x_trafo(flow, xmin=-2, xmax=2, n=25)
    fig.savefig(os.path.join(artifacts_path, "bm_x_trafo.png"))

    fig = plot_chained_bijectors(flow)
    fig.savefig(os.path.join(artifacts_path, "bm_bijectors.png"))

    y = np.linspace(-3, 3, 1000, dtype=np.float32)
    fig = plot_value_and_gradient(flow.bijector.inverse, y.copy())
    fig.savefig(os.path.join(artifacts_path, "bm_bijector.png"))

    def fun(y):
        return flow.bijector.inverse_log_det_jacobian(y, 0)

    fig = plot_value_and_gradient(fun, y.copy())
    fig.savefig(os.path.join(artifacts_path, "bm_ildj.png"))


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run(seed, params, metrics_path, artifacts_path):
    set_seed(seed)
    (train_x, train_y), (val_x, val_y), (test_t, test_y) = prepare_data(
        params.get("data_points", 4000), params.get("scale_data_to_domain", False)
    )
    test_x = np.unique(test_t)

    # Build Model
    model, bf = gen_model(**params["model_kwds"])

    # Fit Model
    hist = fit_model(model, bf, train_x, train_y, val_x, val_y, **params["fit_kwds"])

    if not (
        np.isnan(hist.history["loss"]).any() or np.isnan(hist.history["val_loss"]).any()
    ):
        # Results
        results(
            model,
            bf,
            hist,
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
            test_t,
            metrics_path,
            artifacts_path,
        )

    return model, bf, hist


# %% ifmain
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--10sec", help="run faster", action="store_true", dest="_10sec"
    )
    parser.add_argument(
        "--no-mlflow",
        help="disable mlfow tracking",
        action="store_true",
        default=False,
    )
    parser.add_argument("--seed", help="random seed", default=1, type=int)

    args = parser.parse_args()
    with open("cml/params.yaml") as params_file:
        params = yaml.load(params_file, Loader=yaml.Loader)["bimodal"]

    # Ensure Reproducibility
    print("TFP Version", tfp.__version__)
    print("TF  Version", tf.__version__)

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

    if args._10sec:
        params["fit_kwds"].update({"epochs": 1})
        params["data_points"] = 100

    if USE_MLFLOW and not args.no_mlflow:
        print("mlflow tracking enabled")
        mlflow.autolog()
        exp = mlflow.set_experiment("bernstein_bimodal")
        if os.environ.get("MLFLOW_RUN_ID", False):
            mlflow.start_run()
        with mlflow.start_run(
            experiment_id=exp.experiment_id, nested=mlflow.active_run() is not None
        ):
            mlflow.log_param("seed", args.seed)
            mlflow.log_params(
                dict(filter(lambda kw: not isinstance(kw[1], dict), params.items()))
            )
            mlflow.log_params(params["fit_kwds"])
            run(args.seed, params, metrics_path, artifacts_path)
            mlflow.log_artifacts(artifacts_path)
    else:
        run(args.seed, params, metrics_path, artifacts_path)
