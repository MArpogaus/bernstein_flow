#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-10 15:39:04 (Marcel Arpogaus)
# changed : 2022-05-24 12:41:04 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################

import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util, tensor_util


def get_thetas_constrain_fn(
    support=(-3, 3),
    fn=tf.abs,
    allow_values_outside_support=False,
    constrain_second_drivative=False,
):
    """Ensures monotone increasing Bernstein coefficients.

    :param support: Support of the base distribution. Parameters are scaled to this range.
    :param fn: Function to ensure positive values.
    :param allow_values_outside_support: Use the first and last parameter to
      increase/decrease the upper/lower bound.
    :param constrain_second_drivative: Apply extra contain w.r.t. second derivative ot the polynomial.
                                       Posible values are:
                                        - None: dont apply any constrain
                                        - zero: force second derivative to be zero on boundaries
                                        - turn: force a turning-point on the boundaries
    :returns:   Monotone increasing Bernstein coefficients.
    :rtype:     Tensor

    """

    def constrain_fn(diff):
        dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)
        diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)
        eps = dtype_util.eps(diff.dtype)

        if support and len(support) == 2:
            low, high = support

            low = tensor_util.convert_nonref_to_tensor(low, name="low", dtype=dtype)
            high = tensor_util.convert_nonref_to_tensor(high, name="high", dtype=dtype)

            if allow_values_outside_support:
                low -= fn(diff[..., :1], name="low")
                high += fn(diff[..., -1:], name="high")
                diff = diff[..., 1:-1]

            theta0 = low * tf.ones_like(diff[..., :1])
        else:
            theta0 = diff[..., :1]
            diff = diff[..., 1:]

        diff_positive = tf.math.maximum(fn(diff, name="diff_positive"), eps)

        if constrain_second_drivative:
            if constrain_second_drivative == "zero":
                low2 = diff_positive[..., :1]
                high2 = diff_positive[..., -1:]
            elif constrain_second_drivative == "turn":
                diff_positive = diff_positive[..., 1:-1]
                low2 = diff_positive[..., :1] + fn(diff[..., :1])
                high2 = diff_positive[..., -1:] + fn(diff[..., -1:])
            else:
                raise ValueError(
                    f'Unsupported value "{constrain_second_drivative}" for constrain_second_drivative'
                )
            diff_positive = tf.concat(
                (
                    low2,
                    diff_positive,
                    high2,
                ),
                axis=-1,
            )

        if support:
            diff_positive /= tf.math.reduce_sum(diff_positive, axis=-1)[..., None]
            diff_positive *= high - low
        tc = tf.concat(
            (
                theta0,
                diff_positive,
            ),
            axis=-1,
        )
        return tf.cumsum(tc, axis=-1, name="theta")

    return constrain_fn
