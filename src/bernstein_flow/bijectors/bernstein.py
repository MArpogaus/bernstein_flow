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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)

from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.math.bernstein import (
    gen_bernstein_polynomial_with_linear_extension,
    gen_bernstein_polynomial_with_linear_extrapolation,
)


def reshape_output(batch_shape, sample_shape, y):
    output_shape = prefer_static.broadcast_shape(sample_shape, batch_shape)
    return tf.reshape(y, output_shape)


class BernsteinBijector(tfb.AutoCompositeTensorBijector):
    """
    Implementing Bernstein polynomials using the `tfb.Bijector` interface for
    transformations of a `Distribution` sample.
    """

    def __init__(
        self,
        thetas: tf.Tensor,
        extrapolation: str = False,
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
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([thetas], dtype_hint=tf.float32)

            self.thetas = tensor_util.convert_nonref_to_tensor(
                thetas, name="thetas", dtype=dtype
            )

            if extrapolation:
                (
                    self.b_poly,
                    self.b_poly_log_det_jacobian,
                    self.extra_inv,
                    self.order,
                ) = gen_bernstein_polynomial_with_linear_extrapolation(self.thetas)
            else:
                (
                    self.b_poly,
                    self.b_poly_log_det_jacobian,
                    self.extra_inv,
                    self.order,
                ) = gen_bernstein_polynomial_with_linear_extension(self.thetas)

            self.max_iterations = 50

            super().__init__(
                forward_min_event_ndims=0,
                name=name,
                dtype=dtype,
                parameters=parameters,
                **kwds,
            )

    @classmethod
    def _parameter_properties(cls, dtype=None):
        return dict(
            thetas=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=get_thetas_constrain_fn, event_ndims=1
            ),
        )

    def _apply_fn_and_reshape_output(self, x, fn):
        x = tensor_util.convert_nonref_to_tensor(x, name="x", dtype=self.dtype)

        theta_shape = prefer_static.shape(self.thetas)
        batch_shape = theta_shape[:-1]
        sample_shape = prefer_static.shape(x)

        output = fn(x)

        return reshape_output(batch_shape, sample_shape, output)

    def _forward(self, x):
        return self._apply_fn_and_reshape_output(x, self.b_poly)

    def _forward_log_det_jacobian(self, x):
        return self._apply_fn_and_reshape_output(x, self.b_poly_log_det_jacobian)

    def _inverse(self, y):
        return self._apply_fn_and_reshape_output(y, self._inverse_root_solver)

    def _inverse_root_solver(self, y):
        # Taken from tfp.experimental.bijectors.ScalarFunctionWithInferredInverse
        # Root search inside Bernstein domain [0, 1]
        x, _, num_iterations = tfp.math.find_root_chandrupatla(
            lambda ux: (self._forward(ux) - y),
            low=tf.cast(0, dtype=self.dtype),
            high=tf.cast(1, dtype=self.dtype),
            position_tolerance=dtype_util.eps(self.dtype),
            value_tolerance=dtype_util.eps(self.dtype),
            max_iterations=self.max_iterations,
        )

        x = tf.where(num_iterations < self.max_iterations, x, np.nan)

        # apply inverse extrapolations
        extra_inv = self.extra_inv(y)

        return tf.where(tf.math.is_nan(extra_inv), x, extra_inv)

    def _is_increasing(self, **kwargs):
        return tf.reduce_all(self.thetas[..., 1:] >= self.thetas[..., :-1])
