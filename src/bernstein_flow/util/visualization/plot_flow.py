# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : plot_flow.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-07-12 14:49:21 (Marcel Arpogaus)
# changed : 2024-07-18 12:02:58 (Marcel Arpogaus)

# %% License ###################################################################
# Copyright 2024 Marcel Arpogaus
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

# %% Description ###############################################################
"""Convenience Function to plot a normalizing flow."""

# %% imports ###################################################################
from functools import partial
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from matplotlib.text import Annotation
from tensorflow_probability import bijectors as tfb

from bernstein_flow.bijectors import BernsteinPolynomial

np.random.seed(42)
tf.random.set_seed(42)


def _get_annot_map(bijector_names: List[str], bijector_name: str) -> Dict[str, str]:
    """Get a map from bijector names to annotations.

    Parameters
    ----------
    bijector_names
        List of bijector names.
    bijector_name
        Name of the bijector to split the data at.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping bijector names to annotations.

    """
    annot_map = {}
    for cnt, name in enumerate(bijector_names, 1):
        if name == "sigmoid":
            annot_map[name] = r"$\sigma$"
        else:
            annot_map[name] = f"$f_{{{cnt}}}$"
    annot_map.update({"bijector": annot_map[bijector_name]})
    return annot_map


FORMULAS = {
    tfb.Scale: r"z_{{{curr}}} & = a_{{{curr}}}(\mathbf{{x}}) \cdot z_{{{prev}}}",
    tfb.Shift: r"z_{{{curr}}} & = z_{{{prev}}} + b_{{{curr}}}(\mathbf{{x}})",
    BernsteinPolynomial: r"z_{{{curr}}} & = \frac{{1}}{{M+1}} \sum_{{i=0}}^{{M}}"
    + r"\operatorname{{Be}}_{{i}}^M(z_{{{prev}}}) \vartheta_{{i}}(\mathbf{{x}})",
}


def _get_formulas(bijectors: List[tfb.Bijector]) -> str:
    """Generate LaTeX formulas for a list of bijectors.

    Parameters
    ----------
    bijectors
        List of bijectors.

    Returns
    -------
    str
        LaTeX formulas string.

    """
    formulas = r"\begin{align*}"
    for i, b in enumerate(reversed(bijectors)):
        if b.__class__ in FORMULAS:
            formulas += (
                rf"  f_{{{i + 1}}}: "
                + FORMULAS[b.__class__].format(curr=i + 1, prev=i)
                + r"\\"
            )
    formulas += r"\end{align*}"
    return formulas


def _get_bijectors_recursive(bijector: tfb.Bijector) -> List[tfb.Bijector]:
    """Recursively extract bijectors from a (potentially chained) bijector.

    Parameters
    ----------
    bijector
        The bijector to extract from.

    Returns
    -------
    List[tfb.Bijector]
        List of extracted bijectors.

    """
    if hasattr(bijector, "bijectors"):
        return sum([_get_bijectors_recursive(b) for b in bijector.bijectors], [])
    elif hasattr(bijector, "bijector"):
        return _get_bijectors_recursive(bijector.bijector)
    else:
        return [bijector]


def _get_bijectors(
    flow: tfp.distributions.TransformedDistribution,
) -> List[tfb.Bijector]:
    """Extract bijectors from a transformed distribution.

    Parameters
    ----------
    flow
        Transformed distribution.

    Returns
    -------
    List[tfb.Bijector]
        List of bijectors.

    """
    return _get_bijectors_recursive(flow.bijector)


def _get_bijector_names(bijectors: List[tfb.Bijector]) -> List[str]:
    """Get the names of a list of bijectors.

    Parameters
    ----------
    bijectors
        List of bijectors.

    Returns
    -------
    List[str]
        List of bijector names.

    """
    return [b.name for b in reversed(bijectors)]


def _split_bijector_names(
    bijector_names: List[str], split_bijector_name: str
) -> Tuple[List[str], List[str]]:
    """Split a list of bijector names at a given bijector name.

    Parameters
    ----------
    bijector_names
        List of bijector names.
    split_bijector_name
        Name of the bijector to split the list at.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple containing the two split lists.

    """
    split_index = bijector_names.index(split_bijector_name) + 1
    return bijector_names[:split_index], bijector_names[split_index:]


def _get_plot_data(
    flow: tfp.distributions.TransformedDistribution,
    bijector_name: str,
    n: int,
    z_values: np.ndarray,
    seed: int,
    ignore_bijectors: Tuple[str, ...],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str], List[str]]:
    """Generate plot data for a transformed distribution.

    Parameters
    ----------
    flow
        Transformed distribution.
    bijector_name
        Name of the bijector to split the data at.
    n
        Number of samples
    z_values
        Predefined sample values
    seed
        Random seed
    ignore_bijectors
        Tuple containing names of bijectors to ignore during plotting

    Returns
    -------
    Tuple[Dict[str, Dict[str, np.ndarray]], List[str], List[str]]
        Tuple containing the plot data, post-split bijector names,
        and pre-split bijector names.

    """
    tf.random.set_seed(seed)

    bijectors = _get_bijectors(flow)
    bijector_names = _get_bijector_names(bijectors)
    pre_bpoly_trafos, post_bpoly_trafos = _split_bijector_names(
        bijector_names, bijector_name
    )
    pre_bpoly_trafos = [t for t in pre_bpoly_trafos if t not in ignore_bijectors]
    post_bpoly_trafos = [t for t in post_bpoly_trafos if t not in ignore_bijectors]

    base_dist = flow.distribution

    if z_values is None:
        z_values = base_dist.sample(n)
    z_sorted = np.sort(z_values)

    log_probs = base_dist.log_prob(z_sorted).numpy()

    plot_data: Dict[str, Dict[str, np.ndarray]] = {
        "distribution": {"z": z_sorted, "p": np.exp(log_probs)}
    }
    z = z_sorted[..., None]
    ildj = 0.0

    for i, b in enumerate(bijectors):
        z = b.inverse(z).numpy()
        ildj += b.forward_log_det_jacobian(z, 1)
        name = b.name
        if name not in ignore_bijectors:
            plot_data[name] = {"z": z, "p": np.exp(log_probs + ildj)}

    after_bpoly = next(
        (name for name in post_bpoly_trafos if name in plot_data), "distribution"
    )
    plot_data["bijector"] = {
        "z1": plot_data[bijector_name]["z"],
        "z2": plot_data[after_bpoly]["z"],
    }
    return plot_data, pre_bpoly_trafos, post_bpoly_trafos


def _configure_axes(a: Axes, style: str):
    """Configure the axes of a plot.

    Parameters
    ----------
    a
        Axes object to configure.
    style
        Style of the axes. Can be "right", "top", or "none".

    """
    if style == "right":
        a.spines["top"].set_color("none")
        a.spines["bottom"].set_color("none")
        a.spines["left"].set_color("none")
        a.get_xaxis().set_visible(False)
        a.get_yaxis().tick_right()
        a.get_yaxis().set_label_coords(1, 1.05)
    elif style == "top":
        a.spines["right"].set_color("none")
        a.spines["bottom"].set_color("none")
        a.spines["left"].set_color("none")
        a.get_yaxis().set_visible(False)
        a.get_xaxis().tick_top()
        a.get_xaxis().set_label_coords(1.05, 1)
    elif style == "none":
        a.axis("off")
    a.patch.set_alpha(0.0)


def _prepare_figure(
    plot_data: Dict[str, Dict[str, np.ndarray]],
    pre_bpoly_trafos: List[str],
    post_bpoly_trafos: List[str],
    size: int,
    wspace: float,
    hspace: float,
) -> Tuple[Figure, Dict[str, Axes]]:
    """Prepare the figure and axes for the plot.

    Parameters
    ----------
    plot_data
        Plot data.
    pre_bpoly_trafos
        Pre-split bijector names.
    post_bpoly_trafos
        Post-split bijector names.
    size
        Figure size
    wspace
        Width space between subplots
    hspace
        Height space between subplots

    Returns
    -------
    Tuple[Figure, Dict[str, Axes]]
        Tuple containing the figure and a dictionary mapping bijector names to axes.

    """
    pre_bpoly = sum(k in pre_bpoly_trafos for k in plot_data)
    post_bpoly = sum(k in post_bpoly_trafos for k in plot_data)

    fig = plt.figure(figsize=((pre_bpoly + 1) * size, (post_bpoly + 2) * size))
    gs0 = fig.add_gridspec(
        2,
        2,
        width_ratios=[pre_bpoly / 2, 1],
        height_ratios=[1, (post_bpoly + 1) / 2],
        wspace=wspace,
        hspace=hspace,
    )

    gs00 = gs0[0, 0]
    if pre_bpoly > 1:
        gs00 = gs00.subgridspec(1, pre_bpoly, wspace=wspace)
    else:
        gs00 = [gs00]
    gs00_it = iter(gs00)

    gs11 = gs0[1, 1].subgridspec(post_bpoly + 1, 1, hspace=hspace)
    gs11_it = iter(gs11)

    axs: Dict[str, Axes] = {}
    axs["bijector"] = fig.add_subplot(gs0[0, 1])
    _configure_axes(axs["bijector"], "none")

    idx = 0
    for k in pre_bpoly_trafos + post_bpoly_trafos + ["distribution"]:
        if k not in plot_data:
            continue
        if k in pre_bpoly_trafos:
            axs[k] = fig.add_subplot(next(gs00_it), sharey=axs["bijector"])
            _configure_axes(axs[k], "right")
            set_label = partial(axs[k].set_ylabel, rotation=0, ha="center")
        elif k in post_bpoly_trafos + ["distribution"]:
            axs[k] = fig.add_subplot(next(gs11_it), sharex=axs["bijector"])
            _configure_axes(axs[k], "top")
            set_label = partial(axs[k].set_xlabel, va="center", ha="left")
        set_label("$y$" if idx == 0 else f"$z_{{{idx}}}$")
        idx += 1

    axs["math"] = fig.add_subplot(gs0[1, 0])
    _configure_axes(axs["math"], "none")

    return fig, axs


def _plot_data_to_axes(
    axs: Dict[str, Axes],
    plot_data: Dict[str, Dict[str, np.ndarray]],
    pre_bpoly_trafos: List[str],
    post_bpoly_trafos: List[str],
):
    """Plot the data to the axes.

    Parameters
    ----------
    axs
        Dictionary mapping bijector names to axes.
    plot_data
        Plot data.
    pre_bpoly_trafos
        Pre-split bijector names.
    post_bpoly_trafos
        Post-split bijector names.

    """
    scatter_kwds = dict(c="orange", alpha=0.2, s=8)
    cpd_label = "(transformed) distribution"
    sample_label = "(transformed) samples"
    for k, v in plot_data.items():
        ax = axs[k]
        if k in pre_bpoly_trafos:
            ax.plot(v["p"], v["z"], label=cpd_label)
            m = mpl.markers.MarkerStyle(marker="|")
            m._transform = m.get_transform().rotate_deg(90)
            ax.scatter(
                np.zeros_like(v["z"]),
                v["z"],
                marker=m,
                label=sample_label,
                **scatter_kwds,
            )
            ax.invert_xaxis()
        elif k in post_bpoly_trafos or k == "distribution":
            ax.plot(v["z"], v["p"], label=cpd_label)
            ax.scatter(
                v["z"],
                np.zeros_like(v["z"]),
                marker="|",
                label=sample_label,
                **scatter_kwds,
            )
            ax.invert_yaxis()
        elif k == "bijector":
            ax.scatter(v["z2"], v["z1"], c="orange", s=4)


def _add_annot_to_axes(
    axs: Dict[str, Axes],
    plot_data: Dict[str, Dict[str, np.ndarray]],
    pre_bpoly_trafos: List[str],
    post_bpoly_trafos: List[str],
    bijector_name: str,
    annot_map: Dict[str, str] = {},
    extra_annot_prob: Dict[str, Tuple[Tuple[float, float], str, int]] = {},
    extra_annot_sample: Dict[str, Tuple[Tuple[float, float], str, int]] = {},
    formulas: str = "",
    pos: float = 0.5,
    cp_kwds: Dict = dict(arrowstyle="-|>", shrinkA=10, shrinkB=10, color="gray"),
    usetex: bool = True,
):
    """Add annotations to the axes.

    Parameters
    ----------
    axs
        Dictionary mapping bijector names to axes.
    plot_data
        Plot data.
    pre_bpoly_trafos
        Pre-split bijector names.
    post_bpoly_trafos
        Post-split bijector names.
    bijector_name
        Name of the bijector to split the data at.
    annot_map
        Dictionary mapping bijector names to annotations, by default {}
    extra_annot_prob
        Dictionary containing extra annotations for probabilities, by default {}
    extra_annot_sample
        Dictionary containing extra annotations for samples, by default {}
    formulas
        LaTeX formulas string, by default ""
    pos
        Position of the arrows, by default 0.5
    cp_kwds
        Keyword arguments for the ConnectionPatch,
        by default dict(arrowstyle="-|>", shrinkA=10, shrinkB=10, color="gray")
    usetex
        Whether to use LaTeX for text rendering, by default True

    """
    xyA = None
    axA = None
    z1 = plot_data["bijector"]["z1"]
    z2 = plot_data["bijector"]["z2"]
    i = int(len(z1) * pos)
    z1 = z1[i].item()
    z2 = z2[i].item()
    ax_names = pre_bpoly_trafos + ["bijector"] + post_bpoly_trafos + ["distribution"]
    ax_names = [x for x in ax_names if x in plot_data]
    for kA, kB in zip(ax_names[:-1], ax_names[1:]):
        axA = axs[kA]
        axB = axs[kB]
        if kA == bijector_name:
            xyB = (z2, z1)
            kwdsText = dict(xy=(0.5, 0), ha="center", va="top")
        elif kA in pre_bpoly_trafos:
            xyB = (0, z1)
            kwdsText = dict(xy=(0.5, 0), ha="center", va="top")
        elif kA in post_bpoly_trafos + ["bijector", "distribution"]:
            xyB = (z2, 0)
            kwdsText = dict(xy=(0.5, 0.5), ha="left", va="center")

        cp = ConnectionPatch(
            xyA=xyA or xyB,
            xyB=xyB,
            coordsA="data",
            coordsB="data",
            axesA=axA,
            axesB=axB,
            **cp_kwds,
        )
        axB.add_artist(cp)

        tx = Annotation(xycoords=cp, text=annot_map.get(kA, kA), **kwdsText)
        axB.figure.add_artist(tx)
        xyA = xyB

    axs["math"].text(0.5, 0.5, formulas, ha="center", va="center", usetex=usetex)

    common_arrowprops = dict(
        arrowstyle="-",
        shrinkA=5,
        shrinkB=5,
        facecolor="black",
    )

    k = "bijector"
    z1 = plot_data[k]["z1"]
    z2 = plot_data[k]["z2"]
    i = int(len(z1) * 0.9)
    z1 = z1[i].item()
    z2 = z2[i].item()
    axs[k].annotate(
        "Bernstein\nPolynomial",
        xy=(z2, z1),
        xycoords="data",
        xytext=(-20, 20),
        textcoords="offset points",
        ha="right",
        va="center",
        usetex=usetex,
        arrowprops=dict(**common_arrowprops, connectionstyle="arc3,rad=-0.5"),
    )

    for k, (xytext, text, i) in extra_annot_prob.items():
        z = plot_data[k]["z"]
        p = plot_data[k]["p"]
        if k in pre_bpoly_trafos:
            x = p[i].item()
            y = z[i].item()
            connectionstyle = "arc3,rad=-0.5"
        else:
            x = z[i].item()
            y = p[i].item()
            connectionstyle = "arc3,rad=0.5"
        axs[k].annotate(
            text,
            xy=(x, y),
            xycoords="data",
            xytext=xytext,
            textcoords="offset points",
            ha="right",
            va="center",
            usetex=usetex,
            arrowprops=dict(**common_arrowprops, connectionstyle=connectionstyle),
        )

    for k, (xytext, text, i) in extra_annot_sample.items():
        z = plot_data[k]["z"]
        if k in pre_bpoly_trafos:
            x = 0
            y = z[i].item()
            connectionstyle = "arc3,rad=-0.5"
        else:
            x = z[i].item()
            y = 0
            connectionstyle = "arc3,rad=0.5"
        axs[k].annotate(
            text,
            xy=(x, y),
            xycoords="data",
            xytext=xytext,
            textcoords="offset points",
            ha="right",
            va="center",
            usetex=usetex,
            arrowprops=dict(**common_arrowprops, connectionstyle=connectionstyle),
        )


def plot_flow(
    flow: tfp.distributions.TransformedDistribution,
    bijector_name: str = "bernstein_bijector",
    n: int = 500,
    z_values: np.ndarray = None,
    seed: int = 1,
    size: float = 1.5,
    wspace: float = 0.5,
    hspace: float = 0.5,
    usetex: bool = True,
    ignore_bijectors: Tuple[str, ...] = (),
    **kwds,
) -> Figure:
    """Plot a transformed distribution (flow).

    Parameters
    ----------
    flow
        Transformed distribution to plot.
    bijector_name
        Name of the bijector to split the data at, by default "bernstein_bijector"
    n
        Number of samples, by default 500
    z_values
        Predefined sample values, by default None
    seed
        Random seed, by default 1
    size
        Figure size scaling factor, by default 1.5
    wspace
        Width space between subplots, by default 0.5
    hspace
        Height space between subplots, by default 0.5
    usetex
        Whether to use LaTeX for text rendering, by default True
    ignore_bijectors
        Tuple containing names of bijectors to ignore during plotting, by default ()
    **kwds
        Additional keyword arguments passed to add_annot_to_axes.

    Returns
    -------
    Figure
        The generated matplotlib figure.

    Raises
    ------
    AssertionError
        If the flow is not unimodal (batch shape is not [] or [1]).

    """
    if usetex:
        plt.rcParams.update(
            {"text.latex.preamble": r"\usepackage{amsmath}", "text.usetex": True}
        )
    assert flow.batch_shape in ([], [1]), "Only unimodal distributions supported"
    plot_data, pre_bpoly_trafos, post_bpoly_trafos = _get_plot_data(
        flow,
        bijector_name=bijector_name,
        n=n,
        z_values=z_values,
        seed=seed,
        ignore_bijectors=ignore_bijectors,
    )
    fig, axs = _prepare_figure(
        plot_data,
        pre_bpoly_trafos,
        post_bpoly_trafos,
        size=size,
        wspace=wspace,
        hspace=hspace,
    )
    _plot_data_to_axes(axs, plot_data, pre_bpoly_trafos, post_bpoly_trafos)
    bijectors = _get_bijectors(flow)
    bijector_names = pre_bpoly_trafos + post_bpoly_trafos
    add_annot_to_axes_kwds = {
        **dict(
            bijector_name=bijector_name,
            annot_map=_get_annot_map(bijector_names, bijector_name),
            formulas=_get_formulas(bijectors) if usetex else None,
            usetex=usetex,
        ),
        **kwds,
    }
    _add_annot_to_axes(
        axs,
        plot_data,
        pre_bpoly_trafos,
        post_bpoly_trafos,
        **add_annot_to_axes_kwds,
    )
    handles, labels = axs["distribution"].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, -0.01),
        fancybox=False,
        shadow=False,
        frameon=False,
        loc="lower center",
        ncol=2,
        scatterpoints=50,
        scatteryoffsets=[0.5],
    )

    return fig
