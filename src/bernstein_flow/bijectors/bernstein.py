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
from functools import partial

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import (dtype_util, prefer_static,
                                                    tensor_util)

from bernstein_flow.math.bernstein import (
    derive_bpoly, gen_bernstein_polynom,
    gen_bernstein_polynom_with_linear_extrapolation,
    gen_bernstein_polynom_with_quadratic_extrapolation,
    gen_bernstein_polynom_with_qubic_extrapolation, get_end_points)


def reshape_out(batch_shape, sample_shape, y):
    output_shape = prefer_static.broadcast_shape(sample_shape, batch_shape)
    return tf.reshape(y, output_shape)


class BernsteinBijector(tfp.experimental.bijectors.ScalarFunctionWithInferredInverse):
    """
    Implementing Bernstein polynomials using the `tfb.Bijector` interface for
    transformations of a `Distribution` sample.
    """

    def __init__(
        self,
        thetas: tf.Tensor,
        clip_inverse=1.0e-6,
        name: str = "bernstein_bijector",
        **kwds,
    ):
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

            self.thetas = tensor_util.convert_nonref_to_tensor(
                thetas, name="thetas", dtype=dtype
            )
            self.clip_inverse = tensor_util.convert_nonref_to_tensor(
                clip_inverse, dtype=dtype
            )

            theta_shape = prefer_static.shape(self.thetas)
            batch_shape = theta_shape[:-1]

            z_min, z_max = get_end_points(self.thetas)

            b_poly, self.order = gen_bernstein_polynom(self.thetas)

            def b_boly_reshaped(x):
                x = tensor_util.convert_nonref_to_tensor(x, name="x", dtype=dtype)

                sample_shape = prefer_static.shape(x)

                y = b_poly(x)

                return reshape_out(batch_shape, sample_shape, y)

            # clip = 1.0e-9
            domain_constraint_fn = partial(
                tf.clip_by_value,
                clip_value_min=tf.convert_to_tensor(0, dtype=dtype),
                clip_value_max=tf.convert_to_tensor(1, dtype=dtype),
            )

            def root_search_fn(objective_fn, _, max_iterations=None):
                (
                    estimated_root,
                    objective_at_estimated_root,
                    iteration,
                ) = tfp.math.find_root_chandrupatla(
                    objective_fn,
                    low=z_min,
                    high=z_max,
                    position_tolerance=1e-6,
                    # value_tolerance=1e-7,
                    max_iterations=max_iterations,
                )
                return estimated_root, objective_at_estimated_root, iteration

            super().__init__(
                fn=b_boly_reshaped,
                domain_constraint_fn=domain_constraint_fn,
                root_search_fn=root_search_fn,
                max_iterations=50,
                name=name,
                dtype=dtype,
                **kwds,
            )

    def inverse(self, z):
        z = tensor_util.convert_nonref_to_tensor(z, name="z", dtype=self.dtype)
        y = super().inverse(z)
        return tf.clip_by_value(y, self.clip_inverse, 1.0 - self.clip_inverse)

    def _is_increasing(self, **kwargs):
        return tf.reduce_all(self.thetas[..., 1:] >= self.thetas[..., :-1])


class BernsteinBijectorLinearExtrapolate(
    tfp.experimental.bijectors.ScalarFunctionWithInferredInverse
):
    def __init__(
        self,
        thetas: tf.Tensor,
        name: str = "bernstein_bijector_linear_extrapolate",
        **kwds,
    ):
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

            self.thetas = tensor_util.convert_nonref_to_tensor(thetas, dtype=dtype)

            theta_shape = prefer_static.shape(self.thetas)
            batch_shape = theta_shape[:-1]

            b_poly, self.order = gen_bernstein_polynom_with_linear_extrapolation(
                self.thetas
            )

            def b_boly_reshaped(x):
                x = tensor_util.convert_nonref_to_tensor(x, name="x", dtype=dtype)
                sample_shape = prefer_static.shape(x)

                y = b_poly(x)

                return reshape_out(batch_shape, sample_shape, y)

            def root_search_fn(objective_fn, _, max_iterations=None):
                (
                    estimated_root,
                    objective_at_estimated_root,
                    iteration,
                ) = tfp.math.find_root_chandrupatla(
                    objective_fn,
                    low=tf.ones(1, dtype=dtype),
                    # position_tolerance=tf.convert_to_tensor(1e-6, dtype=dtype),
                    # value_tolerance=1e-7,
                    max_iterations=max_iterations,
                )
                return estimated_root, objective_at_estimated_root, iteration

            super().__init__(
                fn=b_boly_reshaped,
                root_search_fn=root_search_fn,
                max_iterations=50,
                name=name,
                dtype=dtype,
                **kwds,
            )

    def _is_increasing(self, **kwargs):
        return tf.reduce_all(self.thetas[..., 1:] >= self.thetas[..., :-1])


class BernsteinBijectorQuadraticExtrapolate(
    tfp.experimental.bijectors.ScalarFunctionWithInferredInverse
):
    def __init__(
        self,
        thetas: tf.Tensor,
        name: str = "bernstein_bijector_quadratic_extrapolate",
        **kwds,
    ):
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

            self.thetas = tensor_util.convert_nonref_to_tensor(thetas, dtype=dtype)

            theta_shape = prefer_static.shape(self.thetas)
            batch_shape = theta_shape[:-1]

            b_poly, self.order = gen_bernstein_polynom_with_quadratic_extrapolation(
                self.thetas
            )

            def b_boly_reshaped(x):
                x = tensor_util.convert_nonref_to_tensor(x, name="x", dtype=dtype)
                sample_shape = prefer_static.shape(x)

                y = b_poly(x)

                return reshape_out(batch_shape, sample_shape, y)

            def root_search_fn(objective_fn, _, max_iterations=None):
                (
                    estimated_root,
                    objective_at_estimated_root,
                    iteration,
                ) = tfp.math.find_root_chandrupatla(
                    objective_fn,
                    position_tolerance=1e-6,
                    # value_tolerance=1e-7,
                    max_iterations=max_iterations,
                )
                return estimated_root, objective_at_estimated_root, iteration

            super().__init__(
                fn=b_boly_reshaped,
                root_search_fn=root_search_fn,
                max_iterations=50,
                name=name,
                **kwds,
            )

    def _is_increasing(self, **kwargs):
        return tf.reduce_all(self.thetas[..., 1:] >= self.thetas[..., :-1])
