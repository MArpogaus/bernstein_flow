# AUTHOR INFORMATION ##########################################################
# file    : test_bernstein_bijector.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-10-16 08:12:04
# changed : 2020-11-23 18:03:28
# DESCRIPTION #################################################################
#
# This project is following the PEP8 style guide:
#
#    https://www.python.org/dev/peps/pep-0008/)
#
# LICENSE #####################################################################
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
from tensorflow_probability.python.internal import test_util

from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.bijectors import BernsteinBijector


@test_util.test_all_tf_execution_regimes
class BernsteinBijectorTest(tf.test.TestCase):
    def f(self, batch_shape=[], x_shape=[100], order=10, dtype=tf.float32):
        tf.random.set_seed(42)

        thetas_constrain_fn = get_thetas_constrain_fn()
        thetas = thetas_constrain_fn(tf.ones(batch_shape + [order], dtype=dtype))
        eps = 1e-2
        x = tf.random.uniform(x_shape, eps, 1.0 - eps, dtype=dtype)

        bb = BernsteinBijector(thetas=thetas)

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

    def test_inverse_float32(self):
        self.f(batch_shape=[], x_shape=[100], order=10, dtype=tf.float32)

    def test_inverse_batched_float32(self):
        self.f(batch_shape=[2], x_shape=[100, 2], order=10, dtype=tf.float32)

    def test_inverse_batched_multi_float32(self):
        self.f(batch_shape=[2, 4], x_shape=[100, 2, 4], order=10, dtype=tf.float32)

    def test_inverse_batched_multi_huge_float32(self):
        self.f(batch_shape=[16, 48], x_shape=[100, 16, 48], order=10, dtype=tf.float32)

    def test_inverse_float64(self):
        self.f(batch_shape=[], x_shape=[100], order=10, dtype=tf.float64)

    def test_inverse_batched_float64(self):
        self.f(batch_shape=[2], x_shape=[100, 2], order=10, dtype=tf.float64)

    def test_inverse_batched_multi_float64(self):
        self.f(batch_shape=[2, 4], x_shape=[100, 2, 4], order=10, dtype=tf.float64)

    def test_inverse_batched_multi_huge_float64(self):
        self.f(batch_shape=[16, 48], x_shape=[100, 16, 48], order=10, dtype=tf.float64)
