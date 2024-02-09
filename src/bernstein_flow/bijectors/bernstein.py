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
from tensorflow_probability.python.internal import dtype_util, tensor_util

from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.math.bernstein import (
    gen_bernstein_polynomial_with_linear_extension,
    gen_bernstein_polynomial_with_linear_extrapolation,
)


class BernsteinBijector(tfp.experimental.bijectors.ScalarFunctionWithInferredInverse):
    """
    Implementing Bernstein polynomials using the `tfb.Bijector` interface for
    transformations of a `Distribution` sample.
    """

    def __init__(
        self,
        thetas: tf.Tensor,
        extrapolation: str = True,
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

            if extrapolation:
                (
                    b_poly,
                    self._forward_log_det_jacobian,
                    self.b_poly_inverse_extra,
                    self.order,
                ) = gen_bernstein_polynomial_with_linear_extrapolation(self.thetas)
            else:
                (
                    b_poly,
                    self._forward_log_det_jacobian,
                    self.b_poly_inverse_extra,
                    self.order,
                ) = gen_bernstein_polynomial_with_linear_extension(self.thetas)

            domain_constraint_fn = partial(
                tf.clip_by_value, clip_value_min=0.0, clip_value_max=1.0
            )

            def root_search_fn(objective_fn, _, max_iterations=None):
                (
                    estimated_root,
                    objective_at_estimated_root,
                    iteration,
                ) = tfp.math.find_root_chandrupatla(
                    objective_fn,
                    low=tf.convert_to_tensor(0, dtype=dtype),
                    high=tf.convert_to_tensor(1, dtype=dtype),
                    position_tolerance=1e-6,  # dtype_util.eps(dtype),
                    # value_tolerance=1e-7,
                    max_iterations=max_iterations,
                )
                return estimated_root, objective_at_estimated_root, iteration

            super().__init__(
                fn=b_poly,
                domain_constraint_fn=domain_constraint_fn,
                root_search_fn=root_search_fn,
                max_iterations=50,
                name=name,
                dtype=dtype,
                **kwds,
            )

    def _inverse_no_gradient(self, y):
        return tf.stop_gradient(
            self.b_poly_inverse_extra(y, inverse_approx_fn=super()._inverse_no_gradient)
        )

    @classmethod
    def _parameter_properties(cls, dtype=None):
        return dict(
            thetas=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=get_thetas_constrain_fn, event_ndims=1
            ),
        )

    def _is_increasing(self, **kwargs):
        return tf.reduce_all(self.thetas[..., 1:] >= self.thetas[..., :-1])
