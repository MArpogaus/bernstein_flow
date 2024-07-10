#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION #####################################################
# file    : old_faithful.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-03-22 11:14:00 (Marcel Arpogaus)
# changed : 2024-07-10 10:22:54 (Marcel Arpogaus)
# DESCRIPTION ############################################################
# ...
# LICENSE ################################################################
# ...
##########################################################################
# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.distributions import BernsteinFlow
from bernstein_flow.util.visualization import plot_flow
from tensorflow.keras.layers import Dense, InputLayer

# %% Ensure Reproducibility
np.random.seed(2)
tf.random.set_seed(2)
print("TFP Version", tfp.__version__)
print("TF  Version", tf.__version__)


# %% Function Definitions
def bnf(pv):
    return BernsteinFlow.new(
        pv,
        scale_data=True,
        shift_data=True,
        scale_base_distribution=False,
        bernstein_bijector_kwargs=dict(extrapolation=True),
    )


def negloglik(y_true, pv):
    nll = -bnf(pv).log_prob(y_true)
    return nll


# %% Data extracted from
# https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html.
#
# Reference:  A. Azzalini and A. W. Bowman, “A Look at Some Data on the
# Old Faithful Geyser,” Journal of the Royal Statistical Society. Series C
# (Applied Statistics), vol. 39, no. 3, pp. 357–365, 1990, doi:
# 10.2307/2347385.
# fmt: off
y = np.asarray(
    (
        0.6694, 0.3583, 0.6667, 0.6667, 0.6667, 0.3333, 0.7306, 0.7139, 0.3389,
        0.8056, 0.3056, 0.9083, 0.2694, 0.8111, 0.7306, 0.2944, 0.7778, 0.3333,
        0.7889, 0.7028, 0.3167, 0.8278, 0.3333, 0.6667, 0.3333, 0.6667, 0.4722,
        0.75, 0.6778, 0.6194, 0.5861, 0.7444, 0.3694, 0.8139, 0.4333, 0.6917,
        0.3667, 0.7944, 0.3056, 0.7667, 0.3778, 0.6889, 0.3333, 0.6667, 0.3333,
        0.6667, 0.3139, 0.7111, 0.3472, 0.7444, 0.4167, 0.6667, 0.2944, 0.7222,
        0.3639, 0.7472, 0.6472, 0.5556, 0.6222, 0.6667, 0.325, 0.8778, 0.3333,
        0.6667, 0.3333, 0.6667, 0.3333, 0.6667, 0.5889, 0.3611, 0.75, 0.3361,
        0.6917, 0.7, 0.7222, 0.3222, 0.775, 0.6361, 0.6722, 0.6944, 0.7778,
        0.3028, 0.6667, 0.5, 0.6667, 0.3333, 0.7417, 0.3417, 0.7083, 0.3194,
        0.7778, 0.2889, 0.7306, 0.2944, 0.7667, 0.3111, 0.7417, 0.2722, 0.8389,
        0.3028, 0.85, 0.2722, 0.7139, 0.3333, 0.6667, 0.3333, 0.7556, 0.3333,
        0.6667, 0.4889, 0.7889, 0.65, 0.325, 0.6861, 0.3, 0.7778, 0.3056, 0.7833,
        0.3528, 0.7972, 0.3028, 0.6833, 0.775, 0.6667, 0.3333, 0.6667, 0.6667,
        0.7028, 0.6889, 0.6556, 0.625, 0.7361, 0.4111, 0.6944, 0.6333, 0.7194,
        0.6444, 0.7806, 0.2833, 0.8278, 0.7111, 0.7639, 0.6667, 0.6667, 0.6667,
        0.6667, 0.3306, 0.7667, 0.1389, 0.8194, 0.2889, 0.7639, 0.2833, 0.7917,
        0.3056, 0.75, 0.3111, 0.7417, 0.7417, 0.6667, 0.8, 0.6667, 0.6667,
        0.3333, 0.6667, 0.3222, 0.7639, 0.3333, 0.6167, 0.4778, 0.8056, 0.575,
        0.7306, 0.3, 0.7333, 0.4139, 0.7528, 0.35, 0.725, 0.7278, 0.2972,
        0.8194, 0.3028, 0.6667, 0.6667, 0.6667, 0.6444, 0.3083, 0.7833, 0.3361,
        0.7444, 0.3111, 0.6944, 0.3167, 0.7083, 0.5417, 0.7028, 0.3139, 0.8306,
        0.3083, 0.6667, 0.3278, 0.7944, 0.6667, 0.3333, 0.6667, 0.6667, 0.3972,
        0.7361, 0.7028, 0.7278, 0.3333, 0.7417, 0.2917, 0.75, 0.2694, 0.7833,
        0.4278, 0.6167, 0.7056, 0.3222, 0.725, 0.6667, 0.6667, 0.6667, 0.7028,
        0.6667, 0.6889, 0.3139, 0.7444, 0.325, 0.7028, 0.2861, 0.7417, 0.7083,
        0.6611, 0.7306, 0.3278, 0.7417, 0.7111, 0.3194, 0.7361, 0.5, 0.6667,
        0.3333, 0.6667, 0.5472, 0.3056, 0.7694, 0.3056, 0.7694, 0.7667, 0.7083,
        0.3222, 0.8306, 0.3278, 0.7167, 0.7, 0.7556, 0.7333, 0.7694, 0.3333,
        0.6667, 0.6667, 0.6528, 0.3333, 0.75, 0.3, 0.6667, 0.4583, 0.7889,
        0.6611, 0.325, 0.8278, 0.3083, 0.8, 0.6667, 0.6667, 0.6667, 0.6667,
        0.6667, 0.6667, 0.6667, 0.3333, 0.6667, 0.3222, 0.7222, 0.2778, 0.7944,
        0.325, 0.7806, 0.3222, 0.7361, 0.3556, 0.6806, 0.3444, 0.6667, 0.6667,
        0.3333,
    ),
    np.float32,
)
# fmt: on


x = np.ones((y.shape[0], 1))  # We us ones to mimic unconditional data


# %% TensorFlow Dataset API


dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=len(y))
dataset = dataset.batch(32)
dataset = dataset.prefetch(1)
dataset


# %% Fitting the Normalizing Flow to the data


bernstein_order = 9


flow_model = tf.keras.Sequential()
flow_model.add(InputLayer(input_shape=(1)))
# Here could come a gigantus network
flow_model.add(Dense(3 + bernstein_order))

flow_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)

hist = flow_model.fit(dataset, epochs=1000, shuffle=True)


# %% Result
metrcis_path = "metrics/old_faithful/"
artifacts_path = "artifacts/old_faithful/"

os.makedirs(metrcis_path, exist_ok=True)
os.makedirs(artifacts_path, exist_ok=True)

hist_df = pd.DataFrame(hist.history)
hist_df.to_csv(artifacts_path + "of_hist.csv")

with open(metrcis_path + "of_metrics.txt", "w") as metrics:
    metrics.write("Min of loss: " + str(hist_df.loss.min()) + "\n")

fig = hist_df.loss.plot(figsize=(16, 8)).get_figure()
fig.savefig(artifacts_path + "of_hist.png")

flow = bnf(flow_model(np.ones((1, 1), dtype="float32")))

fig = plot_flow(flow)
fig.savefig(artifacts_path + "of_flow.png")

times = np.linspace(0, 1.2)
fp = flow.prob(times)

fig = plt.figure(figsize=(16, 16))
plt.hist(y, 20, density=True)
plt.plot(times, fp)
fig.savefig(artifacts_path + "of_dist.png")


thetas = flow.bijector.bijector.bijectors[0].thetas
a1 = flow.bijector.bijector.bijectors[1].scale
b1 = flow.bijector.bijector.bijectors[2].shift


with open(artifacts_path + "of_pvector.txt", "w") as pvector:
    pvector.write(
        f"""
    a1 = {repr(a1.numpy().flatten())}
    b1 = {repr(b1.numpy().flatten())}
    thetas = {repr(thetas.numpy().flatten())}
"""
    )
