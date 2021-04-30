#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : bernstein_bijector.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-09-11 14:14:24
# changed : 2020-12-07 16:29:11
# DESCRIPTION #################################################################
#
# This project is following the PEP8 style guide:
#
#    https://www.python.org/dev/peps/pep-0008/)
#
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
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import prefer_static

from functools import partial


def gen_beta_dist_h(order, dtype=tf.float32):
    return tfd.Beta(
        tf.range(1, order + 1, dtype=dtype), tf.range(order, 0, -1, dtype=dtype)
    )


def gen_beta_dist_h_dash(order, dtype=tf.float32):
    return tfd.Beta(
        tf.range(1, order, dtype=dtype), tf.range(order - 1, 0, -1, dtype=dtype)
    )


def reshape_out(batch_shape, sample_shape, y):
    output_shape = prefer_static.broadcast_shape(sample_shape, batch_shape)
    return tf.reshape(y, output_shape)


def bernstein_polynom(theta):
    theta_shape = prefer_static.shape(theta)
    order = theta_shape[-1]
    batch_shape = theta_shape[:-1]

    beta_dist_h = gen_beta_dist_h(order, theta.dtype)
    beta_dist_h_dash = gen_beta_dist_h_dash(order, theta.dtype)

    def b_poly(y):
        y = tf.clip_by_value(y, 0, 1.0)

        sample_shape = prefer_static.shape(y)

        y = y[..., tf.newaxis]
        by = beta_dist_h.prob(y)
        z = tf.reduce_mean(by * theta, axis=-1)

        return reshape_out(batch_shape, sample_shape, z)

    def grad(y):
        y = tf.clip_by_value(y, 0, 1.0)

        sample_shape = prefer_static.shape(y)

        y = y[..., tf.newaxis]
        by = beta_dist_h_dash.prob(y)
        dtheta = theta[..., 1:] - theta[..., 0:-1]
        ldj = tf.math.log(tf.reduce_sum(by * dtheta, axis=-1))

        return reshape_out(batch_shape, sample_shape, ldj)

    return b_poly, grad


def constrain_thetas(
    thetas_unconstrained: tf.Tensor,
    low,
    high,
    allow_values_outside_support=False,
    eps=1e-8,
    fn=tf.math.softmax,
) -> tf.Tensor:
    """Ensures monotone increasing Bernstein coefficients.

    :param thetas_unconstrained: Tensor containing the distance of the
      Bernstein coefficients.
    :param low: The lower bound.
    :param high: The upper bound.
    :param allow_values_outside_support: Use the first and last parameter to
      increase/decrease the upper/lower bound.
    :param eps: Optional minimum distance of thetas. Default Value: 1e-8
    :param fn: Function to ensure positive values.
    :returns:   Moncton increasing Bernstein coefficients.
    :rtype:     Tensor

    """
    with tf.name_scope("constrain_theta"):
        if allow_values_outside_support:
            low -= fn(thetas_unconstrained[..., :1], name="low") + eps
            high += fn(thetas_unconstrained[..., -1:], name="high") + eps
            d = fn(thetas_unconstrained[..., 1:-1]) + eps
        else:
            d = fn(thetas_unconstrained) + eps
        d /= tf.reduce_sum(d, axis=-1)[..., None]
        d *= high - low
        tc = tf.concat(
            (
                low * tf.ones_like(thetas_unconstrained[..., :1]),
                d,
            ),
            axis=-1,
        )
        return tf.cumsum(tc, axis=-1, name="theta")


class BernsteinBijector(tfp.experimental.bijectors.ScalarFunctionWithInferredInverse):
    """
    Implementing Bernstein polynomials using the `tfb.Bijector` interface for
    transformations of a `Distribution` sample.
    """

    def __init__(self, thetas: tf.Tensor, name: str = "bernstein_bijector", **kwds):
        """
        Constructs a new instance of a Bernstein polynomial bijector.

        :param      theta:          The Bernstein coefficients.
        :type       theta:          Tensor
        :param      validate_args:  Whether to validate input with asserts.
                                    Passed to `super()`.
        :type       validate_args:  bool
        :param      name:           The name to give Ops created by the
                                    initializer. Passed to `super()`.
        :type       name:           str
        """
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([thetas], dtype_hint=tf.float32)

            self.theta = tensor_util.convert_nonref_to_tensor(thetas, dtype=dtype)

            theta_shape = prefer_static.shape(self.theta)
            self.order = theta_shape[-1]

            self.z_min = tf.math.reduce_min(thetas, axis=-1)
            self.z_max = tf.math.reduce_max(thetas, axis=-1)

            b_poly, grad = bernstein_polynom(thetas)
            self._forward_log_det_jacobian = grad

            # clip = 1.0e-9
            domain_constraint_fn = partial(
                tf.clip_by_value, clip_value_min=0.0, clip_value_max=1.0
            )

            @tf.function
            def root_search_fn(objective_fn, _, max_iterations=None):
                return tfp.math.find_root_chandrupatla(
                    objective_fn,
                    low=self.z_min,
                    high=self.z_max,
                    # position_tolerance=1e-6,
                    # value_tolerance=1e-3,
                    max_iterations=max_iterations - 1,
                )

            super().__init__(
                fn=b_poly,
                domain_constraint_fn=domain_constraint_fn,
                root_search_fn=root_search_fn,
                max_iterations=50,
                name=name,
                **kwds
            )

    def inverse(self, z):
        clip = 1.0e-6
        # z=tf.clip_by_value(z, self.z_min + clip, self.z_max-clip)
        y = super().inverse(z)
        return tf.clip_by_value(y, clip, 1.0 - clip)

    def _is_increasing(self, **kwargs):
        return tf.reduce_all(self.theta[..., 1:] >= self.theta[..., :-1])
