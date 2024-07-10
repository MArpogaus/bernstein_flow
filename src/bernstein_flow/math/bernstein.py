# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : bernstein.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-07-10 10:10:18 (Marcel Arpogaus)
# changed : 2024-07-10 10:10:28 (Marcel Arpogaus)

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

# %% Description ###############################################################
"""Mathematical definitions of Bernstein Polynomials."""

# %% Imports ###################################################################
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import dtype_util, prefer_static

# %% Globals ###################################################################
Tensor = tf.Tensor
Distribution = tfd.Distribution


# %% Functions #################################################################
def reshape_output(
    batch_shape: tf.TensorShape, sample_shape: tf.TensorShape, y: Tensor
) -> Tensor:
    """Reshape tensor to output shape.

    Args:
    ----
        batch_shape (tf.TensorShape): The batch shape of the output.
        sample_shape (tf.TensorShape): The sample shape of the output.
        y (Tensor): The tensor to reshape.

    Returns:
    -------
        Tensor: The reshaped tensor.

    """
    output_shape = prefer_static.broadcast_shape(sample_shape, batch_shape)
    return tf.reshape(y, output_shape)


def gen_basis(order: int, dtype: tf.DType = tf.float32) -> Distribution:
    """Generate Bernstein basis polynomials from Beta distributions.

    Args:
    ----
        order (int): The order of the Bernstein polynomial.
        dtype (tf.DType, optional): The dtype of the Beta distribution.

    Returns:
    -------
        Distribution: A Beta distribution.

    """
    return tfd.Beta(
        tf.range(1, order + 2, dtype=dtype), tf.range(order + 1, 0, -1, dtype=dtype)
    )


def gen_bernstein_polynomial(thetas: Tensor) -> Tuple[Callable[[Tensor], Tensor], int]:
    """Generate Bernstein polynomial as a Callable.

    Args:
    ----
        thetas (Tensor): The weights of the Bernstein polynomial.

    Returns:
    -------
        Callable[[Tensor], Tensor]: A function that evaluates the Bernstein polynomial.
        int: The order of the polynomial.

    """
    theta_shape = prefer_static.shape(thetas)
    order = theta_shape[-1] - 1

    basis = gen_basis(order, thetas.dtype)

    def b_poly(y: Tensor) -> Tensor:
        """Evaluate the Bernstein polynomial.

        Args:
        ----
            y (Tensor): The input to the Bernstein polynomial.

        Returns:
        -------
            Tensor: The output of the Bernstein polynomial.

        """
        y = y[..., tf.newaxis]
        by = basis.prob(y)
        z = tf.reduce_mean(by * thetas, axis=-1)
        return z

    return b_poly, order


def derive_thetas(thetas: Tensor) -> Tensor:
    """Calculate the derivative of the Bernstein polynomial weights.

    Args:
    ----
        thetas (Tensor): The Bernstein polynomial weights.

    Returns:
    -------
        Tensor: The derivative of the Bernstein polynomial weights.

    """
    theta_shape = prefer_static.shape(thetas)
    order = theta_shape[-1] - 1

    dtheta = tf.cast(order, thetas.dtype) * (thetas[..., 1:] - thetas[..., :-1])
    return dtheta


def derive_bpoly(thetas: Tensor) -> Tuple[Callable[[Tensor], Tensor], int]:
    """Generate the derivative of the Bernstein polynomial function.

    Args:
    ----
        thetas (Tensor): The Bernstein polynomial weights.

    Returns:
    -------
        Callable[[Tensor], Tensor]: A function that evaluates the derivative of the
                                    Bernstein polynomial.
        int: The order of the polynomial.

    """
    dtheta = derive_thetas(thetas)
    b_poly_dash, order = gen_bernstein_polynomial(dtheta)
    return b_poly_dash, order


def get_bounds(thetas: Tensor) -> Tensor:
    """Get the bounds of the Bernstein polynomial.

    Args:
    ----
        thetas (Tensor): The Bernstein polynomial weights.

    Returns:
    -------
        Tensor: A tensor containing the lower and upper bounds.

    """
    eps = dtype_util.eps(thetas.dtype)
    x = tf.cast([eps, 1 - eps], dtype=thetas.dtype)

    # adds singleton dimensions for batch shape
    batch_shape = prefer_static.shape(thetas)[:-1]
    batch_rank = prefer_static.rank(batch_shape)

    shape = [...] + [tf.newaxis for _ in range(batch_rank + 1)]
    x = x[shape]

    return x


def evaluate_bpoly_on_bounds(thetas: Tensor, bounds: Tensor) -> Tensor:
    """Evaluate the Bernstein polynomial on the given bounds.

    Args:
    ----
        thetas (Tensor): The Bernstein polynomial weights.
        bounds (Tensor): The bounds to evaluate the polynomial on.

    Returns:
    -------
        Tensor: The Bernstein polynomial evaluated at the given bounds.

    """
    b_poly, _ = gen_bernstein_polynomial(thetas)

    return b_poly(bounds)


def gen_linear_extension(
    thetas: Tensor,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Tensor,
    Tensor,
]:
    """Generate a linear extension function.

    Args:
    ----
        thetas (Tensor): The Bernstein polynomial weights.

    Returns:
    -------
        Callable[[Tensor], Tensor]: The linear extension function.
        Callable[[Tensor], Tensor]: The log determinant Jacobian of the
                                    extension function.
        Callable[[Tensor], Tensor]: The inverse of the extension function.
        Tensor: The x bounds.
        Tensor: The y bounds.

    """
    # [eps, 1 - eps]
    x_bounds = get_bounds(thetas)

    # [Be(eps), Be(1 - eps)]
    y_bounds = evaluate_bpoly_on_bounds(thetas, x_bounds)

    def extra(x: Tensor) -> Tensor:
        """Linear extension function.

        Args:
        ----
            x (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The extended output.

        """
        e0 = x + y_bounds[0]
        e1 = x + y_bounds[1] - 1

        y = tf.where(x <= x_bounds[0], e0, np.nan)
        y = tf.where(x >= x_bounds[1], e1, y)

        return y

    def extra_log_det_jacobian(x: Tensor) -> Tensor:
        """Log determinant Jacobian of the linear extension function.

        Args:
        ----
            x (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The log determinant Jacobian.

        """
        y = tf.where(x <= x_bounds[0], tf.ones_like(x), np.nan)
        y = tf.where(x >= x_bounds[1], tf.ones_like(x), y)

        return tf.math.log(tf.abs(y))

    def extra_inv(y: Tensor) -> Tensor:
        """Inverse of the linear extension function.

        Args:
        ----
            y (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The inverse transformed tensor.

        """
        x0 = y - y_bounds[0]
        x1 = y - y_bounds[1] + 1

        x = tf.where(x0 <= x_bounds[0], x0, np.nan)
        x = tf.where(x1 >= x_bounds[1], x1, x)

        return x

    return extra, extra_log_det_jacobian, extra_inv, x_bounds, y_bounds


def gen_linear_extrapolation(
    thetas: Tensor,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Tensor,
    Tensor,
]:
    """Generate a linear extrapolation function.

    Args:
    ----
        thetas (Tensor): The Bernstein polynomial weights.

    Returns:
    -------
        Callable[[Tensor], Tensor]: The linear extrapolation function.
        Callable[[Tensor], Tensor]: The log determinant Jacobian of the
                                    extrapolation function.
        Callable[[Tensor], Tensor]: The inverse of the extrapolation function.
        Tensor: The x bounds.
        Tensor: The y bounds.

    """
    # [eps, 1 - eps]
    x_bounds = get_bounds(thetas)

    # [Be(eps), Be(1 - eps)]
    y_bounds = evaluate_bpoly_on_bounds(thetas, x_bounds)

    # [Be'(eps), Be'(1 - eps)]
    dtheta = derive_thetas(thetas)
    a = evaluate_bpoly_on_bounds(dtheta, x_bounds)

    def extra(x: Tensor) -> Tensor:
        """Linear extrapolation function.

        Args:
        ----
            x (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The extrapolated output.

        """
        e0 = a[0] * x + y_bounds[0]
        e1 = a[1] * (x - 1) + y_bounds[1]

        y = tf.where(x <= x_bounds[0], e0, np.nan)
        y = tf.where(x >= x_bounds[1], e1, y)

        return y

    def extra_log_det_jacobian(x: Tensor) -> Tensor:
        """Log determinant Jacobian of the linear extrapolation function.

        Args:
        ----
                x (Tensor): The input tensor.

        Returns:
        -------
                Tensor: The log determinant Jacobian.

        """
        y = tf.where(x <= x_bounds[0], a[0], np.nan)
        y = tf.where(x >= x_bounds[1], a[1], y)

        return tf.math.log(tf.abs(y))

    def extra_inv(y: Tensor) -> Tensor:
        """Inverse of the linear extrapolation function.

        Args:
        ----
            y (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The inverse transformed tensor.

        """
        x0 = (y - y_bounds[0]) / a[0]
        x1 = (y - y_bounds[1]) / a[1] + 1

        x = tf.where(x0 <= x_bounds[0], x0, np.nan)
        x = tf.where(x1 >= x_bounds[1], x1, x)

        return x

    return extra, extra_log_det_jacobian, extra_inv, x_bounds, y_bounds


def transform_to_bernstein_domain(x: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """Transform the input to the Bernstein polynomial domain.

    Args:
    ----
        x (Tensor): The input.
        low (Tensor): The lower bound of the domain.
        high (Tensor): The upper bound of the domain.

    Returns:
    -------
        Tensor: The transformed input.

    """
    return (x - low) / (high - low)


def transform_to_support(y: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """Transform the output from the Bernstein polynomial domain to the original domain.

    Args:
    ----
        y (Tensor): The output.
        low (Tensor): The lower bound of the domain.
        high (Tensor): The upper bound of the domain.

    Returns:
    -------
        Tensor: The transformed output.

    """
    return y * (high - low) + low


def gen_bernstein_polynomial_with_extrapolation(
    theta: Tensor,
    gen_extrapolation_fn: Callable = gen_linear_extrapolation,
    domain: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Callable], Tensor],
    int,
]:
    """Generate a Bernstein polynomial with extrapolation.

    Args:
    ----
        theta (Tensor): The Bernstein polynomial weights.
        gen_extrapolation_fn (Callable): The function used to generate the
                                         extrapolation function.
        domain (Optional[Tuple[Tensor, Tensor]]): The domain of the Bernstein polynomial

    Returns:
    -------
        Callable[[Tensor], Tensor]: A function that evaluates the Bernstein polynomial
                                    with extrapolation.
        Callable[[Tensor], Tensor]: A function that computes the log determinant
                                    Jacobian of the function.
        Callable[[Tensor, Callable], Tensor]: A function that computes the inverse of
                                              the function.
        int: The order of the polynomial.

    """
    theta_shape = prefer_static.shape(theta)
    batch_shape = theta_shape[:-1]

    bpoly, order = gen_bernstein_polynomial(theta)
    dbpoly, _ = derive_bpoly(theta)
    extra, extra_log_det_jacobian, extra_inv, x_bounds, y_bounds = gen_extrapolation_fn(
        theta
    )

    def bpoly_extra(x: Tensor) -> Tensor:
        """Bernstein polynomial with extrapolation function.

        Args:
        ----
            x (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The evaluated Bernstein polynomial with extrapolation.

        """
        sample_shape = prefer_static.shape(x)
        x_safe = (x > x_bounds[0]) & (x < x_bounds[1])
        y = bpoly(tf.where(x_safe, x, tf.cast(0.5, theta.dtype)))
        y = tf.where(x_safe, y, extra(x))
        return reshape_output(batch_shape, sample_shape, y)

    def bpoly_log_det_jacobian_extra(x: Tensor) -> Tensor:
        """Log determinant Jacobian of the Bernstein polynomial with extrapolation.

        Args:
        ----
            x (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The log determinant Jacobian.

        """
        sample_shape = prefer_static.shape(x)
        x_safe = (x > x_bounds[0]) & (x < x_bounds[1])
        y = tf.math.log(tf.abs(dbpoly(tf.where(x_safe, x, tf.cast(0.5, theta.dtype)))))
        y = tf.where(x_safe, y, extra_log_det_jacobian(x))
        return reshape_output(batch_shape, sample_shape, y)

    def bpoly_inverse_extra(y: Tensor, inverse_approx_fn: Callable) -> Tensor:
        """Inverse of the Bernstein polynomial with extrapolation.

        Args:
        ----
            y (Tensor): The input tensor.
            inverse_approx_fn (Callable): Function to approximate the inverse.

        Returns:
        -------
            Tensor: The inverse transformed tensor.

        """
        sample_shape = prefer_static.shape(y)
        y_safe = (y > y_bounds[0]) & (y < y_bounds[1])
        x = inverse_approx_fn(tf.where(y_safe, y, tf.cast(0.5, theta.dtype)))
        x = tf.where(y_safe, x, extra_inv(y))
        return reshape_output(batch_shape, sample_shape, x)

    if domain is not None:
        low = tf.convert_to_tensor(domain[0], theta.dtype)
        high = tf.convert_to_tensor(domain[1], theta.dtype)

        def bpoly_extra_scaled(x: Tensor) -> Tensor:
            """Scaled Bernstein polynomial with extrapolation.

            Args:
            ----
                x (Tensor): The input tensor.

            Returns:
            -------
                Tensor: The scaled evaluated Bernstein polynomial with extrapolation.

            """
            x = transform_to_bernstein_domain(x, low, high)
            return bpoly_extra(x)

        def bpoly_log_det_jacobian_extra_scaled(x: Tensor) -> Tensor:
            """Log det. Jacobian of the scaled Bernstein polynomial with extrapolation.

            Args:
            ----
                x (Tensor): The input tensor.

            Returns:
            -------
                Tensor: The log determinant Jacobian.

            """
            x = transform_to_bernstein_domain(x, low, high)
            return bpoly_log_det_jacobian_extra(x) - tf.math.log(high - low)

        def bpoly_inverse_extra_scaled(
            y: Tensor, inverse_approx_fn: Callable
        ) -> Tensor:
            """Inverse of the scaled Bernstein polynomial with extrapolation function.

            Args:
            ----
                y (Tensor): The input tensor.
                inverse_approx_fn (Callable): Function to approximate the inverse.

            Returns:
            -------
                Tensor: The inverse transformed tensor.

            """
            sample_shape = prefer_static.shape(y)
            y_safe = (y > y_bounds[0]) & (y < y_bounds[1])
            x = inverse_approx_fn(tf.where(y_safe, y, tf.cast(0.5, theta.dtype)))
            x_extra = extra_inv(y)
            x = tf.where(y_safe, x, transform_to_support(x_extra, low, high))
            return reshape_output(batch_shape, sample_shape, x)

        return (
            bpoly_extra_scaled,
            bpoly_log_det_jacobian_extra_scaled,
            bpoly_inverse_extra_scaled,
            order,
        )

    else:
        return bpoly_extra, bpoly_log_det_jacobian_extra, bpoly_inverse_extra, order


def gen_bernstein_polynomial_with_linear_extension(*args, **kwargs):
    """Generate a Bernstein polynomial with linear extension.

    Args:
    ----
        *args: Arguments passed to `gen_bernstein_polynomial_with_extrapolation`.
        **kwargs: Keyword arguments passed to
                  `gen_bernstein_polynomial_with_extrapolation`.

    Returns:
    -------
        Tuple: The output of `gen_bernstein_polynomial_with_extrapolation` with
        `gen_extrapolation_fn` set to `gen_linear_extension`.

    """
    return gen_bernstein_polynomial_with_extrapolation(
        *args, gen_extrapolation_fn=gen_linear_extension, **kwargs
    )


def gen_bernstein_polynomial_with_linear_extrapolation(*args, **kwargs):
    """Generate a Bernstein polynomial with linear extrapolation.

    Args:
    ----
        *args: Arguments passed to `gen_bernstein_polynomial_with_extrapolation`.
        **kwargs: Keyword arguments passed to
                  `gen_bernstein_polynomial_with_extrapolation`.

    Returns:
    -------
        Tuple: The output of `gen_bernstein_polynomial_with_extrapolation` with
        `gen_extrapolation_fn` set to `gen_linear_extrapolation`.

    """
    return gen_bernstein_polynomial_with_extrapolation(
        *args, gen_extrapolation_fn=gen_linear_extrapolation, **kwargs
    )
