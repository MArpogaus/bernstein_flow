# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : __init__.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-07-12 15:12:18 (Marcel Arpogaus)
# changed : 2024-07-18 13:00:56 (Marcel Arpogaus)

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
"""Activation functions applied to unconstrained outputs of neural networks."""

# %% imports ###################################################################
from typing import Tuple, Union

import tensorflow as tf
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)


# %% functions #################################################################
def get_thetas_constrain_fn(
    bounds: Tuple[Union[float, None], Union[float, None]] = (-4, 4),
    smooth_bounds: bool = True,
    allow_flexible_bounds: bool = False,
    fn=tf.math.softplus,
    eps: float = 1e-5,
):
    """Return a function that constrains the output of a neural network.

    Parameters
    ----------
    bounds
        The lower and upper bounds of the output.
    smooth_bounds
        Whether to ensure smooth the bounds by enforcing `Be''(0)==Be(1)==0`.
    allow_flexible_bounds
        Whether to allow the bounds to be flexible, i.e. to depend on the input.
    fn
        The positive definite function to apply to the unconstrained parameters.
    eps
        A small number to add to the output to avoid numerical issues.

    Returns
    -------
    callable
        A function that constrains the output of a neural network.

    """
    low, high = bounds

    # @tf.function
    def constrain_fn(diff: tf.Tensor):
        dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)
        diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)

        if low is not None:
            low_theta = tensor_util.convert_nonref_to_tensor(
                low, name="low", dtype=dtype
            ) * tf.ones_like(diff[..., :1])

            if allow_flexible_bounds:
                low_theta -= fn(diff[..., :1])
                diff = diff[..., 1:]
        else:
            low_theta = diff[..., :1]
            diff = diff[..., 1:]

        if high is not None:
            high_theta = tensor_util.convert_nonref_to_tensor(
                high, name="high", dtype=dtype
            )

            if allow_flexible_bounds:
                high_theta += fn(diff[..., :1])
                diff = diff[..., 1:]

        diff_positive = fn(diff)
        if smooth_bounds:
            diff_positive = tf.concat(
                (
                    diff_positive[..., :1],
                    diff_positive,
                    diff_positive[..., -1:],
                ),
                axis=-1,
            )

        if high is not None:
            diff_positive /= tf.reduce_sum(diff_positive, -1)[..., None]
            diff_positive *= (
                high_theta
                - low_theta
                - tf.cast(prefer_static.dimension_size(diff_positive, -1) + 1, dtype)
                * eps
            )
            # diff_positive += eps

        c = tf.concat(
            (
                low_theta,
                diff_positive + eps,
            ),
            axis=-1,
        )
        thetas_constrained = tf.cumsum(c, axis=-1, name="theta")
        return thetas_constrained

    return constrain_fn
