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
from bernstein_flow.math.bernstein import (
    gen_bernstein_polynomial,
    gen_bernstein_polynomial_with_linear_extension,
    gen_bernstein_polynomial_with_linear_extrapolation,
    gen_bernstein_polynomial_with_quadratic_extrapolation,
    get_end_points,
)
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)


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
        extrapolation: str = None,
        name: str = "bernstein_bijector",
        **kwds,
    ) -> None:
        """Constructs a new instance of a Bernstein polynomial bijector.

        :param thetas: The Bernstein coefficients.
        :type thetas: tf.Tensor
        :param extrapolation: Method to extrapolate outside of bounds.
        :type extrapolation: str
        :param name: The name to give Ops created by the initializer.

        """
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([thetas], dtype_hint=tf.float32)

            self.thetas = tensor_util.convert_nonref_to_tensor(
                thetas, name="thetas", dtype=dtype
            )

            theta_shape = prefer_static.shape(self.thetas)
            batch_shape = theta_shape[:-1]

            if extrapolation == "linear":
                b_poly, self.order = gen_bernstein_polynomial_with_linear_extrapolation(
                    self.thetas
                )
            elif extrapolation == "quadratic":
                (
                    b_poly,
                    self.order,
                ) = gen_bernstein_polynomial_with_quadratic_extrapolation(self.thetas)
            else:
                b_poly, self.order = gen_bernstein_polynomial_with_linear_extension(
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
                    # position_tolerance=1e-6,
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
