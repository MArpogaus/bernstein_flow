#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : visualization.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-13 16:04:37
# COPYRIGHT ###################################################################
# Copyright 2020 Marcel Arpogaus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTES ######################################################################
#
# This project is following the
# [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-10-14 20:10:44
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-13 16:04:37
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

import numpy as np


# function definitions ########################################################
def vizualize_flow_from_z_domain(flow):
    bijector = flow.bijector
    base_dist = flow.distribution

    n = 1000

    # z ~ p_z
    z_samples = np.linspace(-3, 3, n, dtype=np.float32)  # flow.sample(n)
    z_probs = base_dist.prob(z_samples)

    # y = h(z)
    y_samples = np.squeeze(bijector.forward(z_samples))

    # p_y(y) = p_z(h^-1(y))*|h^-1'(y)| = p_z(z)*|h^-1'(y)|
    ildj = bijector.inverse_log_det_jacobian(y_samples, 0)
    log_prob = base_dist.log_prob(z_samples)
    log_prob = log_prob + ildj

    y_probs = np.exp(log_prob)

    fig = plt.figure(figsize=(8, 8))

    gs1 = gridspec.GridSpec(2, 2)

    b_ax = fig.add_subplot(gs1[0, 1])
    b_ax.spines['top'].set_color('none')  # don't draw spine
    b_ax.spines['right'].set_color('none')  # don't draw spine
    b_ax.spines['bottom'].set_position(('outward', 10))
    b_ax.spines['left'].set_position(('outward', 10))

    b_ax.set_xlabel('z')
    b_ax.set_ylabel('y')

    y_ax = fig.add_subplot(gs1[0, 0], sharey=b_ax)
    y_ax.axis('off')
    plt.gca().invert_xaxis()

    z_ax = fig.add_subplot(gs1[1, 1], sharex=b_ax)
    z_ax.axis('off')
    plt.gca().invert_yaxis()

    gs1.tight_layout(fig)

    z_ax.plot(z_samples, z_probs)
    b_ax.plot(z_samples, y_samples)
    y_ax.plot(y_probs, y_samples)

    mu_z = base_dist.mean()

    mu_y = bijector.forward(mu_z)

    p_mu_z = base_dist.prob(mu_z)
    p_mu_y = np.exp(base_dist.log_prob(mu_z) +
                    bijector.inverse_log_det_jacobian(mu_y, 0))

    cp_kwds = dict(
        color='darkgray',
        lw=1,
        ls='--',
        arrowstyle='->'
    )

    project_y = ConnectionPatch(
        xyA=(mu_z, p_mu_z),
        xyB=(mu_z, mu_y),
        coordsA='data',
        coordsB='data',
        axesA=z_ax,
        axesB=b_ax,
        **cp_kwds
    )
    z_ax.add_artist(project_y)

    project_y = ConnectionPatch(
        xyA=(mu_z, mu_y),
        xyB=(p_mu_y, mu_y),
        coordsA='data',
        coordsB='data',
        axesA=b_ax,
        axesB=y_ax,
        **cp_kwds
    )
    b_ax.add_artist(project_y)

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
    ildj = 0.
    for i, (a, b) in enumerate(zip(ax[1:], chained_bijectors)):
        # we need to use the inverse here since we are going from z->y!
        z = b.inverse(zz)
        ildj += b.forward_log_det_jacobian(z, 1)
        # print(z.shape, zz.shape, ildj.shape)
        a.plot(z, np.exp(log_probs + ildj))
        a.set_title(b.name.replace('_', ' '))
        a.set_xlabel(f'$z_{i}$')
        a.set_ylabel(f'$p(z_{i+1})$')
        zz = z
    fig.tight_layout()


def plot_flow(flow, y, ax=plt, color='skyblue'):
    base_dist = flow.distribution
    bijector = flow.bijector

    dense_y = flow.prob(y).numpy()

    mu = np.squeeze(bijector.forward(base_dist.mean()))
    plus_sd = np.squeeze(bijector.forward(
        base_dist.mean() + base_dist.stddev()))
    minus_sd = np.squeeze(bijector.forward(
        base_dist.mean() - base_dist.stddev()))

    ax.plot(
        [mu, mu],
        [np.min(dense_y),
         flow.prob(mu.reshape(-1, 1)).numpy()],
        color='black',
        lw=2
    )
    ax.plot(
        [plus_sd, plus_sd],
        [np.min(dense_y), flow.prob(plus_sd.reshape(-1, 1)).numpy()],
        '--',
        color='green'
    )
    ax.plot(
        [minus_sd, minus_sd],
        [np.min(dense_y), flow.prob(minus_sd.reshape(-1, 1)).numpy()],
        '--',
        color='green'
    )

    def quant(p):
        q = bijector.forward(base_dist.quantile(p))
        return np.squeeze(q)

    qs = [.05, .1, .2, .3, .4]
    ax.fill_between(
        np.squeeze(y),
        np.squeeze(dense_y),
        np.min(dense_y),
        fc=color,
        alpha=max(qs)
    )
    for i, q in enumerate(sorted(qs)):
        ax.fill_between(
            np.squeeze(y),
            np.squeeze(dense_y),
            np.min(dense_y),
            where=(
                (np.squeeze(y) > quant(q)) & (np.squeeze(y) < quant(1 - q))
            ),
            fc=color,
            alpha=q / max(qs)
        )

    ax.plot(
        y,
        dense_y,
        '-',
        color=color,
        linewidth=2
    )
