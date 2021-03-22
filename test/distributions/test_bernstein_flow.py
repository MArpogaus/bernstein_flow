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
import random
import numpy as np

import tensorflow as tf

from tensorflow.python.framework import random_seed

from tensorflow_probability import distributions as tfd

from bernstein_flow.distributions import BernsteinFlow

# Python RNG
random.seed(42)

# Numpy RNG
np.random.seed(42)

# TF RNG
random_seed.set_seed(42)


class BernsteinFlowTest(tf.test.TestCase):
    def gen_pvs(self, batch_shape, order):
        return tf.random.uniform(
            shape=batch_shape + [4 + order], minval=-1000, maxval=100
        )

    def gen_dist(self, batch_shape, order=5, **kwds):
        if batch_shape != []:
            n = tfd.Normal(loc=tf.zeros((batch_shape)), scale=tf.ones((batch_shape)))
            bs = BernsteinFlow(self.gen_pvs(batch_shape, order), **kwds)
        else:
            n = tfd.Normal(loc=tf.zeros((1)), scale=tf.ones((1)))
            bs = BernsteinFlow(self.gen_pvs(batch_shape, order), **kwds)
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

    def test_log_normal(self):
        batch_shape = [32, 48]
        log_normal = tfd.LogNormal(loc=tf.zeros(batch_shape), scale=1)
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=log_normal,
            scale_base_distribution=False,
        )
        self.f(normal_dist, trans_dist)

    def test_student_t(self):
        batch_shape = [32, 48]
        student_t = tfd.StudentT(2, loc=tf.zeros(batch_shape), scale=1)
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=student_t,
            base_dist_lower_bound=-25,
            base_dist_upper_bound=25,
            scale_base_distribution=False,
        )
        self.f(normal_dist, trans_dist)

    def test_weibull(self):
        batch_shape = [32, 48]
        weibull = tfd.Weibull(0.5, scale=tf.ones(batch_shape))
        normal_dist, trans_dist = self.gen_dist(
            batch_shape=batch_shape,
            base_distribution=weibull,
            scale_base_distribution=False,
        )
        self.f(normal_dist, trans_dist)

    def test_small_numbers(self):
        for o in [5, 20, 2000]:
            bf = BernsteinFlow(
                [1, 1] + 5 * [-1000] + (o - 4) * [1] + 5 * [-1000] + [1, 1, 1],
            )
            n = tfd.Normal(loc=tf.zeros((1)), scale=tf.ones((1)))
            self.f(n, bf)

    def test_random_numbers(self):
        for bs in [[1], [32], [32, 48]]:
            for _ in range(10):
                bf = BernsteinFlow(self.gen_pvs(bs, 50))
                n = tfd.Normal(loc=tf.zeros(bs), scale=tf.ones(bs))
                self.f(n, bf)
