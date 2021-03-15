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

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from bernstein_flow.distributions import BernsteinFlow


class BernsteinFlowTest(tf.test.TestCase):
    def gen_dist(self, batch_shape):
        order = 5
        if batch_shape != []:
            n = tfd.Normal(loc=tf.zeros((batch_shape)), scale=tf.ones((batch_shape)))
            bs = BernsteinFlow(tf.ones(batch_shape + [3 + order]))
        else:
            n = tfd.Normal(loc=tf.zeros((1)), scale=tf.ones((1)))
            bs = BernsteinFlow(tf.ones((3 + order)))
        return n, bs

    def test_dist(self, batch_shape=[]):
        bernstein_order = 5

        normal_dist, trans_dist = self.gen_dist(batch_shape)

        for input_shape in [[1], [1, 1], [1] + batch_shape]:

            # Check the distribution.
            self.assertIsInstance(trans_dist, tfd.TransformedDistribution)
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

    def test_dist_batch(self):
        self.test_dist(batch_shape=[32])

    def test_dist_multi(self):
        self.test_dist(batch_shape=[32, 48])

    def test_small_numbers(self):
        x = tf.linspace(-100.0, 100.0, 10000)
        for o in [5, 20, 2000]:
            bf = BernsteinFlow(
                [1, 1] + 2 * [-1000] + (o - 4) * [1] + 2 * [-1000] + [1],
            )

            self.assertAllInRange(bf.prob(x), bf.dtype.min, bf.dtype.max)
            self.assertAllInRange(bf.sample(100000), bf.dtype.min, bf.dtype.max)
