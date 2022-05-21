#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : bernstein.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-09 08:45:52 (Marcel Arpogaus)
# changed : 2022-05-21 08:36:09 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################
# REQUIRED MODULES ############################################################
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static


def gen_basis(order, dtype=tf.float32):
    return tfd.Beta(
        tf.range(1, order + 1, dtype=dtype), tf.range(order, 0, -1, dtype=dtype)
    )


def gen_bernstein_polynomial(thetas):
    theta_shape = prefer_static.shape(thetas)
    order = theta_shape[-1]

    basis = gen_basis(order, thetas.dtype)

    def b_poly(y):
        y = y[..., tf.newaxis]
        by = basis.prob(y)
        z = tf.reduce_mean(by * thetas, axis=-1)

        return z

    return b_poly, order


def derive_thetas(thetas):
    theta_shape = prefer_static.shape(thetas)
    order = theta_shape[-1]

    dtheta = tf.cast(order - 1, thetas.dtype) * (thetas[..., 1:] - thetas[..., :-1])
    return dtheta, order


def derive_bpoly(thetas):
    dtheta, _ = derive_thetas(thetas)
    b_poly_dash, order = gen_bernstein_polynomial(dtheta)
    return b_poly_dash, order


def get_end_points(thetas):
    return thetas[..., 0], thetas[..., -1]


def gen_linear_extension(thetas):
    # y = x + b
    # y' = a

    # [Be(0),Be(1)]
    end_values = get_end_points(thetas)

    # b = y - a = Be - a
    b = tf.stack(end_values)

    def extra(x):
        y = np.nan * tf.ones_like(x)
        e0 = x + b[0]
        e1 = x + b[1] - 1

        y = tf.where(x <= 0, e0, y)
        y = tf.where(x >= 1, e1, y)

        return y

    return extra


def gen_linear_extrapolation(thetas):
    # y = a * x + b
    # y' = a

    # [Be(0),Be(1)]
    end_values = get_end_points(thetas)

    # [Be'(0),Be'(1)]
    dtheta, order = derive_thetas(thetas)
    first_derivative_end_values = get_end_points(dtheta)

    # a = b' = Be'
    a = tf.stack(first_derivative_end_values)

    # b = y - a = Be - a
    b = tf.stack(end_values) - a

    def extra(x):
        y = np.nan * tf.ones_like(x)
        e0 = a[0] * (x + 1) + b[0]
        e1 = a[1] * x + b[1]

        y = tf.where(x <= 0, e0, y)
        y = tf.where(x >= 1, e1, y)

        return y

    # def extra_inv(y):
    #     x=np.nan*tf.ones_like(y)
    #
    #     x0 = (y - b[0])/ a[0] - 1
    #     x1 = (y - b[1])/ a[1]

    #     x = tf.where(x0 <= 0, x0, x)
    #     x = tf.where(x1 >= 1, x1, x)

    #     return x

    return extra  # , extra_inv


def gen_quadratic_extrapolation(thetas):
    # y= ax^2 + bx + c
    # [Be(0),Be(1)]
    end_values = tf.stack(get_end_points(thetas))

    # [Be'(0),Be'(1)]
    dtheta, _ = derive_thetas(thetas)
    first_derivative_end_values = tf.stack(get_end_points(dtheta))

    # [Be''(0),Be''(1)]
    dtheta, _ = derive_thetas(dtheta)
    second_derivative_end_values = tf.stack(get_end_points(dtheta))

    # a = y''/2
    a = second_derivative_end_values / 2

    # b = y'-2a
    b = first_derivative_end_values - 2 * a

    # c = y - a -b
    c = end_values - a - b

    def extra(x):
        y = np.nan * tf.ones_like(x)
        e0 = a[0] * (x + 1) ** 2 + b[0] * (x + 1) + c[0]
        e1 = a[1] * x**2 + b[1] * x + c[1]

        y = tf.where(x <= 0, e0, y)
        y = tf.where(x >= 1, e1, y)

        return y

    return extra


def gen_qubic_extrapolation(thetas):
    # [b(0),b(1)]
    end_values = get_end_points(thetas)

    # [b'(0),b'(1)]
    dtheta, _ = derive_thetas(thetas)
    first_derivative_end_values = tf.stack(get_end_points(dtheta))

    # [b''(0),b''(1)]
    dtheta, _ = derive_thetas(dtheta)
    second_derivative_end_values = tf.stack(get_end_points(dtheta))

    # [b''(0),b''(1)]
    dtheta, _ = derive_thetas(dtheta)
    third_derivative_end_values = tf.stack(get_end_points(dtheta))

    # a = y'''
    a = third_derivative_end_values

    # b = (y''-6a) / 2
    b = (second_derivative_end_values - 6 * a) / 2

    # c = y' - 3a - 2b
    c = first_derivative_end_values - 3 * a - 2 * b

    # c = y - a - b - c
    d = end_values - a - b - c

    def extra(x):
        y = np.nan * tf.ones_like(x)
        e0 = a[0] * (x + 1) ** 3 + b[0] * (x + 1) ** 2 + c[0] * (x + 1) + d[0]
        e1 = a[1] * x**3 + b[1] * x**2 + c[1] * x + d[1]

        y = tf.where(x <= 0, e0, y)
        y = tf.where(x >= 1, e1, y)

        return y

    return extra


def gen_bernstein_polynomial_with_extrapolation(
    theta, gen_extrapolation_fn=gen_linear_extrapolation
):
    bpoly, order = gen_bernstein_polynomial(theta)
    extra = gen_extrapolation_fn(theta)

    def bpoly_extra(x):
        y = bpoly(tf.where((x <= 0) | (x >= 1), 0.5 * tf.ones_like(x), x))
        y = tf.where((x <= 0) | (x >= 1), extra(x), y)
        return y

    return bpoly_extra, order


def gen_bernstein_polynomial_with_linear_extension(thetas):
    return gen_bernstein_polynomial_with_extrapolation(
        thetas, gen_extrapolation_fn=gen_linear_extension
    )


def gen_bernstein_polynomial_with_linear_extrapolation(thetas):
    return gen_bernstein_polynomial_with_extrapolation(
        thetas, gen_extrapolation_fn=gen_linear_extrapolation
    )


def gen_bernstein_polynomial_with_quadratic_extrapolation(thetas):
    return gen_bernstein_polynomial_with_extrapolation(
        thetas, gen_extrapolation_fn=gen_quadratic_extrapolation
    )


def gen_bernstein_polynomial_with_qubic_extrapolation(thetas):
    return gen_bernstein_polynomial_with_extrapolation(
        thetas, gen_extrapolation_fn=gen_qubic_extrapolation
    )
