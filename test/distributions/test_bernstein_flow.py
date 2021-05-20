#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : test_bernstein_flow.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-10-22 10:46:18
# changed : 2020-11-23 18:03:30
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

from tensorflow_probability import distributions as tfd

from bernstein_flow.distributions import BernsteinFlow
from bernstein_flow.bijectors import BernsteinBijectorLinearExtrapolate

tf.random.set_seed(42)


class BernsteinFlowTest(tf.test.TestCase):
    def gen_pvs(self, batch_shape, order):
        return tf.random.uniform(
            shape=batch_shape + [4 + order], minval=-1000, maxval=100
        )

    def gen_dist(self, batch_shape, order=5, **kwds):
        if batch_shape != []:
            n = tfd.Normal(loc=tf.zeros((batch_shape)), scale=tf.ones((batch_shape)))
            bs = BernsteinFlow.from_pvector(self.gen_pvs(batch_shape, order), **kwds)
        else:
            n = tfd.Normal(loc=tf.zeros((1)), scale=tf.ones((1)))
            bs = BernsteinFlow.from_pvector(self.gen_pvs(batch_shape, order), **kwds)
        return n, bs

    def f(self, normal_dist, trans_dist):

        for input_shape in [[1], [1, 1], [1] + normal_dist.batch_shape]:
            x = tf.random.uniform(
                shape=[1000] + normal_dist.batch_shape, minval=-100, maxval=100
            )

            # Check the distribution type
            self.assertIsInstance(trans_dist, tfd.TransformedDistribution)

            # check the shapes
            self.assertEqual(normal_dist.batch_shape, trans_dist.batch_shape)
            self.assertEqual(normal_dist.event_shape, trans_dist.event_shape)
            self.assertEqual(
                normal_dist.sample(input_shape).shape,
                trans_dist.sample(input_shape).shape,
            )
            self.assertEqual(
                normal_dist.prob(tf.zeros(input_shape)).shape,
                trans_dist.prob(tf.zeros(input_shape)).shape,
            )

            # check the Normalization
            self.assertAllClose(
                normal_dist.cdf(normal_dist.dtype.min),
                trans_dist.cdf(trans_dist.dtype.min),
                atol=1e-3,
            )
            self.assertAllClose(
                normal_dist.cdf(normal_dist.dtype.max),
                trans_dist.cdf(trans_dist.dtype.max),
                atol=1e-3,
            )

            # check for infs and nans
            self.assertAllInRange(trans_dist.prob(x), 0, trans_dist.dtype.max)
            self.assertAllInRange(
                trans_dist.mean(), trans_dist.dtype.min, trans_dist.dtype.max
            )
            self.assertAllInRange(
                trans_dist.sample(10000), trans_dist.dtype.min, trans_dist.dtype.max
            )
            try:
                self.assertAllInRange(
                    trans_dist.quantile(1e-5),
                    trans_dist.dtype.min,
                    trans_dist.dtype.max,
                )
                self.assertAllInRange(
                    trans_dist.quantile(1 - 1e-5),
                    trans_dist.dtype.min,
                    trans_dist.dtype.max,
                )
            except NotImplementedError:
                pass

    def test_dist_batch(self):
        normal_dist, trans_dist = self.gen_dist(batch_shape=[32])
        self.f(normal_dist, trans_dist)

    def test_dist_multi(self):
        normal_dist, trans_dist = self.gen_dist(batch_shape=[32, 48])
        self.f(normal_dist, trans_dist)

    def test_dist_batch_extra(self):
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=[32],
            bb_class=BernsteinBijectorLinearExtrapolate,
            clip_to_bernstein_domain=False,
        )
        self.f(normal_dist, trans_dist)

    def test_dist_multi_extra(self):
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=[32],
            bb_class=BernsteinBijectorLinearExtrapolate,
            clip_to_bernstein_domain=False,
        )
        self.f(normal_dist, trans_dist)

    def test_log_normal(self):
        batch_shape = [32, 48]
        log_normal = tfd.LogNormal(loc=tf.zeros(batch_shape), scale=1)
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=log_normal,
            support=(1e-10, tf.math.exp(4.0)),
            scale_base_distribution=True,
        )
        self.f(normal_dist, trans_dist)

    def test_logistic(self):
        batch_shape = [32, 48]
        logistic = tfd.Logistic(loc=tf.zeros(batch_shape), scale=1)
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=logistic,
            support=(-8, 8),
            scale_base_distribution=True,
            allow_values_outside_support=True,
        )
        self.f(normal_dist, trans_dist)

    def test_uniform(self):
        batch_shape = [32, 48]
        uniform = tfd.Uniform(-tf.ones(batch_shape), tf.ones(batch_shape))
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=uniform,
            support=(-1.0, 1.0),
            scale_base_distribution=False,
        )
        self.f(normal_dist, trans_dist)

    def test_student_t(self):
        batch_shape = [32, 48]
        student_t = tfd.StudentT(2, loc=tf.zeros(batch_shape), scale=1)
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=student_t,
            support=(-25, 25),
            scale_base_distribution=False,
            allow_values_outside_support=True,
        )
        self.f(normal_dist, trans_dist)

    def test_weibull(self):
        batch_shape = [32, 48]
        weibull = tfd.Weibull(0.5, scale=tf.ones(batch_shape))
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=weibull,
            support=(1e-10, 50),
            scale_base_distribution=False,
        )
        self.f(normal_dist, trans_dist)

    def test_small_numbers(self):
        for o in [5, 20, 2000]:
            bf = BernsteinFlow.from_pvector(
                [1, 1] + 5 * [-1000] + (o - 4) * [1] + 5 * [-1000] + [1, 1, 1],
            )
            n = tfd.Normal(loc=0.0, scale=1.0)
            self.f(n, bf)

    def test_random_numbers(self):
        for bs in [[2], [32], [32, 48]]:
            for _ in range(10):
                normal_dist, trans_dist = self.gen_dist(batch_shape=bs)
                self.f(normal_dist, trans_dist)
