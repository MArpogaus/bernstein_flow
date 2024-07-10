# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : test_bernstein_bijector.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-07-10 10:09:06 (Marcel Arpogaus)
# changed : 2024-07-10 10:09:06 (Marcel Arpogaus)

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
"""Unit test for Bernstein polynomial bijector."""

# REQUIRED PYTHON MODULES #####################################################
from functools import partial
from itertools import product

import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.bijectors import BernsteinPolynomial
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class BernsteinBijectorTest(tf.test.TestCase):
    def f(self, batch_shape=[], x_shape=[100], order=10, dtype=tf.float32, **kwds):
        tf.random.set_seed(42)

        thetas_constrain_fn = get_thetas_constrain_fn(smooth_bounds=True)
        thetas = thetas_constrain_fn(tf.ones(batch_shape + [order], dtype=dtype))
        eps = 1e-2
        x = tf.random.uniform(x_shape, eps, 1.0 - eps, dtype=dtype)

        bb = BernsteinPolynomial(thetas=thetas)

        forward_x = bb.forward(x)
        # Use identity to invalidate cache.
        inverse_x = bb.inverse(tf.identity(forward_x))
        forward_inverse_x = bb.forward(inverse_x)

        fldj = bb.forward_log_det_jacobian(x, event_ndims=1)
        # Use identity to invalidate cache.
        ildj = bb.inverse_log_det_jacobian(tf.identity(forward_x), event_ndims=1)

        self.assertAllClose(x, inverse_x, rtol=1e-5, atol=1e-7)
        self.assertAllClose(forward_x, forward_inverse_x, rtol=1e-5, atol=1e-6)
        self.assertAllClose(ildj, -fldj, rtol=1e-5, atol=1e-7)

        # Test on bounds
        shape = tf.concat([[-1], tf.ones(tf.rank(thetas) - 1, dtype=tf.int32)], 0)
        values = tf.range(2, dtype=dtype)  # [0, 1]
        values = tf.reshape(values, shape)
        [value, grad] = tfp.math.value_and_gradient(bb.forward, values)
        self.assertAllInRange(value, thetas.dtype.min, thetas.dtype.max)
        self.assertAllInRange(grad, thetas.dtype.min, thetas.dtype.max)
        [value, grad] = tfp.math.value_and_gradient(bb.inverse, tf.identity(value))
        self.assertAllInRange(value, thetas.dtype.min, thetas.dtype.max)
        self.assertAllInRange(grad, thetas.dtype.min, thetas.dtype.max)
        [value, grad] = tfp.math.value_and_gradient(
            partial(bb.forward_log_det_jacobian, event_ndims=0), values
        )
        self.assertAllInRange(value, thetas.dtype.min, thetas.dtype.max)
        self.assertAllInRange(grad, thetas.dtype.min, thetas.dtype.max)
        [value, grad] = tfp.math.value_and_gradient(
            partial(bb.inverse_log_det_jacobian, event_ndims=0), tf.identity(value)
        )
        self.assertAllInRange(value, thetas.dtype.min, thetas.dtype.max)
        self.assertAllInRange(grad, thetas.dtype.min, thetas.dtype.max)

    def test_inverse_float32(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[],
                x_shape=[100],
                order=10,
                dtype=tf.float32,
                extrapolation=extrapolation,
                domain=domain,
            )

    def test_inverse_batched_float32(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[2],
                x_shape=[100, 2],
                order=10,
                dtype=tf.float32,
                extrapolation=extrapolation,
                domain=domain,
            )

    def test_inverse_batched_multi_float32(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[2, 4],
                x_shape=[100, 2, 4],
                order=10,
                dtype=tf.float32,
                extrapolation=extrapolation,
                domain=domain,
            )

    def test_inverse_batched_multi_huge_float32(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[16, 48],
                x_shape=[100, 16, 48],
                order=10,
                dtype=tf.float32,
                extrapolation=extrapolation,
                domain=domain,
            )

    def test_inverse_float64(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[],
                x_shape=[100],
                order=10,
                dtype=tf.float64,
                extrapolation=extrapolation,
                domain=domain,
            )

    def test_inverse_batched_float64(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[2],
                x_shape=[100, 2],
                order=10,
                dtype=tf.float64,
                extrapolation=extrapolation,
                domain=domain,
            )

    def test_inverse_batched_multi_float64(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[2, 4],
                x_shape=[100, 2, 4],
                order=10,
                dtype=tf.float64,
                extrapolation=extrapolation,
                domain=domain,
            )

    def test_inverse_batched_multi_huge_float64(self):
        for extrapolation, domain in product([True, False], [None, [-4, 5]]):
            self.f(
                batch_shape=[16, 48],
                x_shape=[100, 16, 48],
                order=10,
                dtype=tf.float64,
                extrapolation=extrapolation,
                domain=domain,
            )
