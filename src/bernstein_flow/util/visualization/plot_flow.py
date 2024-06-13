# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : plot_flow.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-06-01 15:21:22 (Marcel Arpogaus)
# changed : 2024-06-13 21:27:07 (Marcel Arpogaus)
# DESCRIPTION #################################################################
#
# This project is following the PEP8 style guide:
#
#    https://www.python.org/dev/peps/pep-0008/
#
# LICENSE #####################################################################
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
###############################################################################
# REQUIRED PYTHON MODULES #####################################################
from functools import partial, reduce
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

from bernstein_flow.bijectors import BernsteinBijector

np.random.seed(42)
tf.random.set_seed(42)


def get_annot_map(bijector_names: List[str], bijector_name: str, cnt: int = 0) -> Dict:
    """Get a map from bijector names to annotations.

    Parameters
    ----------
    bijector_names : List[str]
        List of bijector names.
    bijector_name : str
        Name of the bijector to split the list at.
    cnt : int, optional
        Counter for the annotations, by default 0

    Returns
    -------
    Dict
        Dictionary mapping bijector names to annotations.
    """

    def annot_map_mapper(name: str) -> Tuple[str, str]:
        nonlocal cnt
        if name == "sigmoid":
            annot = r"$\sigma$"
        else:
            cnt += 1
            annot = f"$f_{{{cnt}}}$"
        return (name, annot)

    annot_map = dict(map(annot_map_mapper, bijector_names))
    annot_map.update({"bijector": annot_map[bijector_name]})
    return annot_map


FORMULAS = {
    tfb.Scale: r"z_{{{curr}}} & = a_{{{curr}}}(\mathbf{{x}}) \cdot z_{{{prev}}}",
    tfb.Shift: r"z_{{{curr}}} & = z_{{{prev}}} + b_{{{curr}}}(\mathbf{{x}})",
    BernsteinBijector: r"z_{{{curr}}} & = \frac{{1}}{{M+1}} \sum_{{i=0}}^{{M}}"
    + r"\operatorname{{Be}}_{{i}}^M(z_{{{prev}}}) \vartheta_{{i}}(\mathbf{{x}})",
}


def get_formulas(bijectors: List[tfb.Bijector]) -> str:
    """Generate LaTeX formulas for a list of bijectors.

    Parameters
    ----------
    bijectors : List[tfb.Bijector]
        List of bijectors.

    Returns
    -------
    str
        LaTeX formulas string.
    """
    formuals = r"\begin{align*}"
    cnt = 1
    for b in reversed(bijectors):
        if b.__class__ in FORMULAS.keys():
            formuals += (
                rf"  f_{{{cnt}}}: "
                + FORMULAS[b.__class__].format(curr=cnt, prev=cnt - 1)
                + r"\\"
            )
            cnt += 1
    formuals += r"\end{align*}"
    return formuals


def get_bijectors(
    flow: tfp.distributions.TransformedDistribution,
) -> List[tfb.Bijector]:
    """Extract bijectors from a transformed distribution.

    Parameters
    ----------
    flow : tfp.distributions.TransformedDistribution
        Transformed distribution.

    Returns
    -------
    List[tfb.Bijector]
        List of bijectors.
    """
    return flow.bijector.bijector.bijectors


def get_bijector_names(bijectors: List[tfb.Bijector]) -> List[str]:
    """Get the names of a list of bijectors.

    Parameters
    ----------
    bijectors : List[tfb.Bijector]
        List of bijectors.

    Returns
    -------
    List[str]
        List of bijector names.
    """
    return list(map(lambda x: x.name, reversed(bijectors)))


def split_bijector_names(
    bijector_names: List[str], split_bijector_name: str
) -> Tuple[List[str], List[str]]:
    """Split a list of bijector names at a given bijector name.

    Parameters
    ----------
    bijector_names : List[str]
        List of bijector names.
    split_bijector_name : str
        Name of the bijector to split the list at.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple containing the two split lists.
    """
    bpoly_idx = bijector_names.index(split_bijector_name) + 1
    return bijector_names[:bpoly_idx], bijector_names[bpoly_idx:]


def get_intersec_reducer(arr: List) -> callable:
    """Generate a reducer function to count intersections.

    Parameters
    ----------
    arr : List
        List to check for intersections.

    Returns
    -------
    callable
        Reducer function.
    """

    def reducer(c: int, i: int) -> int:
        if i in arr:
            return c + 1
        else:
            return c

    return reducer


def get_plot_data(
    flow: tfp.distributions.TransformedDistribution,
    bijector_name: str,
    n: int = 200,
    z_values: np.ndarray = None,
    seed: int = 1,
) -> Tuple[Dict, List[str], List[str]]:
    """Generate plot data for a transformed distribution.

    Parameters
    ----------
    flow : tfp.distributions.TransformedDistribution
        Transformed distribution.
    bijector_name : str
        Name of the bijector to split the data at.
    n : int, optional
        Number of samples, by default 200
    z_values : np.ndarray, optional
        Predefined sample values, by default None
    seed : int, optional
        Random seed, by default 1

    Returns
    -------
    Tuple[Dict, List[str], List[str]]
        Tuple containing the plot data, post-split bijector names,
        and pre-split bijector names.
    """
    tf.random.set_seed(seed)

    chained_bijectors = get_bijectors(flow)
    bijector_names = get_bijector_names(chained_bijectors)
    pre_bpoly_trafos, post_bpoly_trafos = split_bijector_names(
        bijector_names, bijector_name
    )

    base_dist = flow.distribution

    if z_values is None:
        z_values = base_dist.sample(n)
    z_sorted = np.sort(z_values)

    log_probs = base_dist.log_prob(z_sorted).numpy()

    zz = z_sorted[..., None]
    ildj = 0.0
    plot_data = {"distribution": dict(z=z_sorted, p=np.exp(log_probs))}

    for i, b in enumerate(chained_bijectors):
        # we need to use the inverse here since we are going from z->y!
        z = b.inverse(zz).numpy()
        ildj += b.forward_log_det_jacobian(z, 1)
        name = b.name
        plot_data[name] = dict(z=z, p=np.exp(log_probs + ildj))
        zz = z

    after_bpoly = next(iter(post_bpoly_trafos), "distribution")
    z1 = plot_data[bijector_name]["z"]
    z2 = plot_data[after_bpoly]["z"]
    plot_data["bijector"] = dict(z1=z1, z2=z2)
    return plot_data, post_bpoly_trafos, pre_bpoly_trafos


def configure_axes(a: Axes, style: str):
    """Configure the axes of a plot.

    Parameters
    ----------
    a : Axes
        Axes object to configure.
    style : str
        Style of the axes. Can be "right", "top", or "none".
    """
    if style == "right":
        a.spines["top"].set_color("none")  # don't draw spine
        # a.spines["right"].set_color("none")  # don't draw spine
        a.spines["bottom"].set_color("none")  # don't draw spine
        a.spines["left"].set_color("none")  # don't draw spine
        a.get_xaxis().set_visible(False)
        a.get_yaxis().tick_right()
        a.get_yaxis().set_label_coords(1, 1.05)
    elif style == "top":
        # a.spines["top"].set_color("none")  # don't draw spine
        a.spines["right"].set_color("none")  # don't draw spine
        a.spines["bottom"].set_color("none")  # don't draw spine
        a.spines["left"].set_color("none")  # don't draw spine
        a.get_yaxis().set_visible(False)
        a.get_xaxis().tick_top()
        a.get_xaxis().set_label_coords(1.05, 1)
    elif style == "none":
        a.axis("off")
    a.patch.set_alpha(0.0)


def prepare_figure(
    plot_data: Dict,
    pre_bpoly_trafos: List[str],
    post_bpoly_trafos: List[str],
    size: int = 4,
    wspace: float = 0.5,
    hspace: float = 0.5,
) -> Tuple[Figure, Dict[str, Axes]]:
    """Prepare the figure and axes for the plot.

    Parameters
    ----------
    plot_data : Dict
        Plot data.
    pre_bpoly_trafos : List[str]
        Pre-split bijector names.
    post_bpoly_trafos : List[str]
        Post-split bijector names.
    size : int, optional
        Figure size, by default 4
    wspace : float, optional
        Width space between subplots, by default 0.5
    hspace : float, optional
        Height space between subplots, by default 0.5

    Returns
    -------
    Tuple[Figure, Dict[str, Axes]]
        Tuple containing the figure and a dictionary mapping bijector names to axes.
    """
    pre_bpoly = reduce(get_intersec_reducer(pre_bpoly_trafos), plot_data.keys(), 0)
    post_bpoly = reduce(get_intersec_reducer(post_bpoly_trafos), plot_data.keys(), 0)

    fig = plt.figure(figsize=((pre_bpoly + 1) * size, (post_bpoly + 2) * size))
    gs0 = fig.add_gridspec(
        2,
        2,
        width_ratios=[pre_bpoly / 2, 1],
        height_ratios=[1, (post_bpoly + 1) / 2],
        wspace=0.2,
        hspace=0.2,
    )

    gs00 = gs0[0, 0]
    if pre_bpoly > 1:
        gs00 = gs00.subgridspec(1, pre_bpoly, wspace=wspace)
    else:
        gs00 = [gs00]
    gs00_it = iter(gs00)

    gs11 = gs0[1, 1].subgridspec(post_bpoly + 1, 1, hspace=hspace)
    gs11_it = iter(gs11)

    axs = {}
    axs["bijector"] = fig.add_subplot(gs0[0, 1])
    configure_axes(axs["bijector"], "none")
    idx = 0
    for k in pre_bpoly_trafos + post_bpoly_trafos + ["distribution"]:
        if k not in plot_data.keys():
            continue
        if k in pre_bpoly_trafos:
            axs[k] = fig.add_subplot(next(gs00_it), sharey=axs["bijector"])
            configure_axes(axs[k], "right")
            set_label = partial(axs[k].set_ylabel, rotation=0, ha="center")
        elif k in post_bpoly_trafos + ["distribution"]:
            axs[k] = fig.add_subplot(next(gs11_it), sharex=axs["bijector"])
            configure_axes(axs[k], "top")
            set_label = partial(axs[k].set_xlabel, va="center", ha="left")
        if idx == 0:
            set_label("$y$")
        else:
            set_label(f"$z_{{{idx}}}$")
        idx += 1

    axs["math"] = fig.add_subplot(gs0[1, 0])
    configure_axes(axs["math"], "none")

    return fig, axs


def plot_data_to_axes(
    axs: Dict[str, Axes],
    plot_data: Dict,
    pre_bpoly_trafos: List[str],
    post_bpoly_trafos: List[str],
):
    """Plot the data to the axes.

    Parameters
    ----------
    axs : Dict[str, Axes]
        Dictionary mapping bijector names to axes.
    plot_data : Dict
        Plot data.
    pre_bpoly_trafos : List[str]
        Pre-split bijector names.
    post_bpoly_trafos : List[str]
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
            ax = axs[k]
            z1 = plot_data[k]["z1"]
            z2 = plot_data[k]["z2"]
            ax.scatter(z2, z1, c="orange", s=4)


def add_annot_to_axes(
    axs: Dict[str, Axes],
    plot_data: Dict,
    pre_bpoly_trafos: List[str],
    post_bpoly_trafos: List[str],
    bijector_name: str,
    annot_map: Dict = {},
    extra_annot_prob: Dict = {},
    extra_annot_sample: Dict = {},
    formulas: str = "",
    pos: float = 0.5,
    cp_kwds: Dict = dict(arrowstyle="-|>", shrinkA=10, shrinkB=10, color="gray"),
    usetex: bool = True,
):
    """Add annotations to the axes.

    Parameters
    ----------
    axs : Dict[str, Axes]
        Dictionary mapping bijector names to axes.
    plot_data : Dict
        Plot data.
    pre_bpoly_trafos : List[str]
        Pre-split bijector names.
    post_bpoly_trafos : List[str]
        Post-split bijector names.
    bijector_name : str
        Name of the bijector to split the data at.
    annot_map : Dict, optional
        Dictionary mapping bijector names to annotations, by default {}
    extra_annot_prob : Dict, optional
        Dictionary containing extra annotations for probabilities, by default {}
    extra_annot_sample : Dict, optional
        Dictionary containing extra annotations for samples, by default {}
    formulas : str, optional
        LaTeX formulas string, by default ""
    pos : float, optional
        Position of the arrows, by default 0.5
    cp_kwds : Dict, optional
        Keyword arguments for the ConnectionPatch,
        by default dict(arrowstyle="-|>", shrinkA=10, shrinkB=10, color="gray")
    usetex : bool, optional
        Whether to use LaTeX for text rendering, by default True
    """
    # add arrows for bijectors
    xyA = None
    axA = None
    z1 = plot_data["bijector"]["z1"]
    z2 = plot_data["bijector"]["z2"]
    i = int(len(z1) * pos)
    z1 = z1[i].item()
    z2 = z2[i].item()
    ax_names = pre_bpoly_trafos + ["bijector"] + post_bpoly_trafos + ["distribution"]
    ax_names = list(filter(lambda x: x in plot_data.keys(), ax_names))
    for kA, kB in zip(ax_names[:-1], ax_names[1:]):
        axA = axs[kA]
        axB = axs[kB]
        if kA == bijector_name:
            xyB = (z2, z1)
            kwdsText = dict(
                xy=(0.5, 0),
                ha="center",
                va="top",
            )
        elif kA in pre_bpoly_trafos:
            xyB = (0, z1)
            kwdsText = dict(
                xy=(0.5, 0),
                ha="center",
                va="top",
            )
        elif kA in post_bpoly_trafos + ["bijector", "distribution"]:
            xyB = (z2, 0)
            kwdsText = dict(
                xy=(0.5, 0.5),
                ha="left",
                va="center",
            )

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

    # add formulas
    ax = axs["math"]
    ax.text(0.5, 0.5, formulas, ha="center", va="center", usetex=usetex)

    common_arrowprops = dict(
        arrowstyle="-",
        shrinkA=5,
        shrinkB=5,
        facecolor="black",
    )

    # annotate Bernstein polinomial
    k = "bijector"
    z1 = plot_data[k]["z1"]
    z2 = plot_data[k]["z2"]
    i = int(len(z1) * 0.9)
    z1 = z1[i].item()
    z2 = z2[i].item()
    ax = axs[k]
    ax.annotate(
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
        arrowprops = common_arrowprops.copy()
        if k in pre_bpoly_trafos:
            x = p[i].item()
            y = z[i].item()
            arrowprops["connectionstyle"] = "arc3,rad=-0.5"
        else:
            x = z[i].item()
            y = p[i].item()
            arrowprops["connectionstyle"] = "arc3,rad=0.5"
        ax = axs[k]
        ax.annotate(
            text,
            xy=(x, y),
            xycoords="data",
            xytext=xytext,
            textcoords="offset points",
            ha="right",
            va="center",
            usetex=usetex,
            arrowprops=arrowprops,
        )

    for k, (xytext, text, i) in extra_annot_sample.items():
        z = plot_data[k]["z"]
        arrowprops = common_arrowprops.copy()
        if k in pre_bpoly_trafos:
            x = 0
            y = z[i].item()
            arrowprops["connectionstyle"] = "arc3,rad=-0.5"
        else:
            x = z[i].item()
            y = 0
            arrowprops["connectionstyle"] = "arc3,rad=0.5"
        ax = axs[k]
        ax.annotate(
            text,
            xy=(x, y),
            xycoords="data",
            xytext=xytext,
            textcoords="offset points",
            ha="right",
            va="center",
            usetex=usetex,
            arrowprops=arrowprops,
        )


def plot_flow(
    flow: tfp.distributions.TransformedDistribution,
    bijector_name: str = "bpoly",
    n: int = 500,
    z_values: np.ndarray = None,
    size: float = 1.5,
    usetex: bool = True,
    **kwds,
) -> Figure:
    """Plot a transformed distribution (flow).

    Parameters
    ----------
    flow : tfp.distributions.TransformedDistribution
        Transformed distribution to plot.
    bijector_name : str, optional
        Name of the bijector to split the data at, by default "bpoly"
    n : int, optional
        Number of samples, by default 500
    z_values : np.ndarray, optional
        Predefined sample values, by default None
    size : float, optional
        Figure size scaling factor, by default 1.5
    usetex : bool, optional
        Whether to use LaTeX for text rendering, by default True
    **kwds : optional
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
            {
                # for the align enivironment
                "text.latex.preamble": r"\usepackage{amsmath}",
                "text.usetex": True,  # use inline math for ticks
            }
        )
    assert flow.batch_shape == [] or flow.batch_shape == [
        1
    ], "Only unimodal distributions supported"
    plot_data, post_bpoly_trafos, pre_bpoly_trafos = get_plot_data(
        flow, bijector_name=bijector_name, n=n, z_values=z_values
    )
    fig, axs = prepare_figure(plot_data, pre_bpoly_trafos, post_bpoly_trafos, size=size)
    plot_data_to_axes(axs, plot_data, pre_bpoly_trafos, post_bpoly_trafos)
    bijectors = get_bijectors(flow)
    add_annot_to_axes_kwds = {
        **dict(
            bijector_name=bijector_name,
            annot_map=get_annot_map(
                get_bijector_names(bijectors), bijector_name=bijector_name
            ),
            formulas=get_formulas(bijectors) if usetex else None,
            usetex=usetex,
        ),
        **kwds,
    }
    add_annot_to_axes(
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
