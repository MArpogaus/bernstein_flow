# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : bernstein.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-09 08:45:52 (Marcel Arpogaus)
# changed : 2024-02-16 16:43:03 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################
# REQUIRED MODULES ############################################################
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import dtype_util, prefer_static


def reshape_output(batch_shape, sample_shape, y):
    output_shape = prefer_static.broadcast_shape(sample_shape, batch_shape)
    return tf.reshape(y, output_shape)


def gen_basis(order, dtype=tf.float32):
    return tfd.Beta(
        tf.range(1, order + 2, dtype=dtype), tf.range(order + 1, 0, -1, dtype=dtype)
    )


def gen_bernstein_polynomial(thetas):
    theta_shape = prefer_static.shape(thetas)
    order = theta_shape[-1] - 1

    basis = gen_basis(order, thetas.dtype)

    def b_poly(y):
        y = y[..., tf.newaxis]
        by = basis.prob(y)
        z = tf.reduce_mean(by * thetas, axis=-1)

        return z

    return b_poly, order


def derive_thetas(thetas):
    theta_shape = prefer_static.shape(thetas)
    order = theta_shape[-1] - 1

    dtheta = tf.cast(order, thetas.dtype) * (thetas[..., 1:] - thetas[..., :-1])
    return dtheta


def derive_bpoly(thetas):
    dtheta = derive_thetas(thetas)
    b_poly_dash, order = gen_bernstein_polynomial(dtheta)
    return b_poly_dash, order


def get_bounds(thetas):
    eps = dtype_util.eps(thetas.dtype)
    x = tf.cast([eps, 1 - eps], dtype=thetas.dtype)

    # adds singleton dimensions for batch shape
    batch_shape = prefer_static.shape(thetas)[:-1]
    batch_rank = prefer_static.rank(batch_shape)

    shape = [...] + [tf.newaxis for _ in range(batch_rank + 1)]
    x = x[shape]

    return x


def evaluate_bpoly_on_bounds(thetas, bounds):
    b_poly, _ = gen_bernstein_polynomial(thetas)

    return b_poly(bounds)


def gen_linear_extension(thetas):
    # y = x + b
    # y' = a

    # [eps, 1 - eps]
    x_bounds = get_bounds(thetas)

    # [Be(eps), Be(1 - eps)]
    y_bounds = evaluate_bpoly_on_bounds(thetas, x_bounds)

    def extra(x):
        e0 = x + y_bounds[0]
        e1 = x + y_bounds[1] - 1

        y = tf.where(x <= x_bounds[0], e0, np.nan)
        y = tf.where(x >= x_bounds[1], e1, y)

        return y

    def extra_log_det_jacobian(x):
        y = tf.where(x <= x_bounds[0], tf.ones_like(x), np.nan)
        y = tf.where(x >= x_bounds[1], tf.ones_like(x), y)

        return tf.math.log(tf.abs(y))

    def extra_inv(y):
        x0 = y - y_bounds[0]
        x1 = y - y_bounds[1] + 1

        x = tf.where(x0 <= x_bounds[0], x0, np.nan)
        x = tf.where(x1 >= x_bounds[1], x1, x)

        return x

    return extra, extra_log_det_jacobian, extra_inv, x_bounds, y_bounds


def gen_linear_extrapolation(thetas):
    # y = a * x + b
    # y' = a

    # [eps, 1 - eps]
    x_bounds = get_bounds(thetas)

    # [Be(eps), Be(1 - eps)]
    y_bounds = evaluate_bpoly_on_bounds(thetas, x_bounds)

    # [Be'(eps), Be'(1 - eps)]
    dtheta = derive_thetas(thetas)
    a = evaluate_bpoly_on_bounds(dtheta, x_bounds)

    def extra(x):
        e0 = a[0] * x + y_bounds[0]
        e1 = a[1] * (x - 1) + y_bounds[1]

        y = tf.where(x <= x_bounds[0], e0, np.nan)
        y = tf.where(x >= x_bounds[1], e1, y)

        return y

    def extra_log_det_jacobian(x):
        y = tf.where(x <= x_bounds[0], a[0], np.nan)
        y = tf.where(x >= x_bounds[1], a[1], y)

        return tf.math.log(tf.abs(y))

    def extra_inv(y):
        x0 = (y - y_bounds[0]) / a[0]
        x1 = (y - y_bounds[1]) / a[1] + 1

        x = tf.where(x0 <= x_bounds[0], x0, np.nan)
        x = tf.where(x1 >= x_bounds[1], x1, x)

        return x

    return extra, extra_log_det_jacobian, extra_inv, x_bounds, y_bounds


def gen_bernstein_polynomial_with_extrapolation(
    theta, gen_extrapolation_fn=gen_linear_extrapolation
):
    theta_shape = prefer_static.shape(theta)
    batch_shape = theta_shape[:-1]

    bpoly, order = gen_bernstein_polynomial(theta)
    dbpoly, _ = derive_bpoly(theta)
    extra, extra_log_det_jacobian, extra_inv, x_bounds, y_bounds = gen_extrapolation_fn(
        theta
    )

    def bpoly_extra(x):
        sample_shape = prefer_static.shape(x)
        x_safe = (x > x_bounds[0]) & (x < x_bounds[1])
        y = bpoly(tf.where(x_safe, x, tf.cast(0.5, theta.dtype)))
        y = tf.where(x_safe, y, extra(x))
        return reshape_output(batch_shape, sample_shape, y)

    def bpoly_log_det_jacobian_extra(x):
        sample_shape = prefer_static.shape(x)
        x_safe = (x > x_bounds[0]) & (x < x_bounds[1])
        y = tf.math.log(tf.abs(dbpoly(tf.where(x_safe, x, tf.cast(0.5, theta.dtype)))))
        y = tf.where(x_safe, y, extra_log_det_jacobian(x))
        return reshape_output(batch_shape, sample_shape, y)

    def bpoly_inverse_extra(y, inverse_approx_fn):
        sample_shape = prefer_static.shape(y)
        y_safe = (y > y_bounds[0]) & (y < y_bounds[1])
        x = inverse_approx_fn(tf.where(y_safe, y, tf.cast(0.5, theta.dtype)))
        x = tf.where(y_safe, x, extra_inv(y))
        return reshape_output(batch_shape, sample_shape, x)

    return bpoly_extra, bpoly_log_det_jacobian_extra, bpoly_inverse_extra, order


def gen_bernstein_polynomial_with_linear_extension(thetas):
    return gen_bernstein_polynomial_with_extrapolation(
        thetas, gen_extrapolation_fn=gen_linear_extension
    )


def gen_bernstein_polynomial_with_linear_extrapolation(thetas):
    return gen_bernstein_polynomial_with_extrapolation(
        thetas, gen_extrapolation_fn=gen_linear_extrapolation
    )
