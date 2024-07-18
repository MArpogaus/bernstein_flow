# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : bernstein.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-07-12 14:52:28 (Marcel Arpogaus)
# changed : 2024-07-18 12:01:50 (Marcel Arpogaus)

# %% License ###################################################################
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
# %% Description ###############################################################
"""Implement a Bernstein Polynomial as a `tfp.Bijector`."""

# %% imports ###################################################################
from functools import partial
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util, tensor_util

from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.math.bernstein import (
    generate_bernstein_polynomial_with_linear_extension,
    generate_bernstein_polynomial_with_linear_extrapolation,
)


# %% classes ###################################################################
class BernsteinPolynomial(tfp.experimental.bijectors.ScalarFunctionWithInferredInverse):
    """Implement a Bernstein polynomials Bijector.

    This bijector transforms an input tensor by applying a Bernstein polynomial.

    Attributes
    ----------
    thetas
        The Bernstein coefficients.

    """

    def __init__(
        self,
        thetas: tf.Tensor,
        extrapolation: bool = True,
        analytic_jacobian: bool = True,
        domain: Tuple[float, float] = (0.0, 1.0),
        name: str = "bernstein_bijector",
        **kwargs,
    ) -> None:
        """Construct a new instance of a Bernstein polynomial bijector.

        Parameters
        ----------
        thetas
            The Bernstein coefficients.
        extrapolation
            The method to extrapolate outside of bounds.
        analytic_jacobian
            Whether to use the analytic Jacobian.
        domain
            The domain of the Bernstein polynomial.
        name
            The name to give Ops created by the initializer.
        kwargs
            Keyword arguments for the parent class.

        """
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([thetas], dtype_hint=tf.float32)

            self.thetas = tensor_util.convert_nonref_to_tensor(
                thetas, name="thetas", dtype=dtype
            )

            if extrapolation:
                (
                    b_poly,
                    forward_log_det_jacobian,
                    self.b_poly_inverse_extra,
                    self.order,
                ) = generate_bernstein_polynomial_with_linear_extrapolation(
                    self.thetas, domain=domain
                )
            else:
                (
                    b_poly,
                    forward_log_det_jacobian,
                    self.b_poly_inverse_extra,
                    self.order,
                ) = generate_bernstein_polynomial_with_linear_extension(
                    self.thetas, domain=domain
                )

            low = tf.convert_to_tensor(domain[0], dtype=dtype)
            high = tf.convert_to_tensor(domain[1], dtype=dtype)

            if analytic_jacobian:
                self._forward_log_det_jacobian = forward_log_det_jacobian

            domain_constraint_fn = partial(
                tf.clip_by_value, clip_value_min=low, clip_value_max=high
            )

            def root_search_fn(objective_fn, _, max_iterations=None):
                (
                    estimated_root,
                    objective_at_estimated_root,
                    iteration,
                ) = tfp.math.find_root_chandrupatla(
                    objective_fn,
                    low=low,
                    high=high,
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
                **kwargs,
            )

    def _inverse_no_gradient(self, y: tf.Tensor) -> tf.Tensor:
        """Compute the inverse of the bijector."""
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

    def _is_increasing(self, **kwargs) -> bool:
        """Check if the bijector is increasing."""
        return tf.reduce_all(self.thetas[..., 1:] >= self.thetas[..., :-1])
