#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-10 15:39:04 (Marcel Arpogaus)
# changed : 2022-09-01 14:34:32 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################

import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util, tensor_util


def get_thetas_constrain_fn(
    low=-4,
    high=4,
    smooth_bounds=False,
    allow_flexible_bounds=False,
    fn=tf.math.softplus,
    eps=1e-5,
):
    def constrain_fn(diff):
        dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)

        diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)
        low_theta = tensor_util.convert_nonref_to_tensor(low, name="low", dtype=dtype)
        high_theta = tensor_util.convert_nonref_to_tensor(
            high, name="high", dtype=dtype
        )

        if allow_flexible_bounds:
            low_theta -= fn(diff[..., :1])
            high_theta += fn(diff[..., -1:])
            diff = diff[..., 1:-1]

        theta0 = low_theta * tf.ones_like(diff[..., :1])
        diff_positive = fn(diff)

        if smooth_bounds:
            low2 = diff_positive[..., :1]
            high2 = diff_positive[..., -1:]
            diff_positive = tf.concat(
                (
                    low2,
                    diff_positive,
                    high2,
                ),
                axis=-1,
            )

        diff_positive /= tf.reduce_sum(diff_positive, -1)[..., None]
        diff_positive *= high_theta - low_theta

        c = tf.concat(
            (
                theta0,
                tf.maximum(diff_positive, eps),
            ),
            axis=-1,
        )
        return tf.cumsum(c, axis=-1, name="theta")

    return constrain_fn
