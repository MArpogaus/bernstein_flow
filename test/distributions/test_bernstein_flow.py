#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : test_bernstein_flow.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-10-22 10:46:18
# changed : 2020-10-23 11:25:34
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
import tensorflow.keras.layers as tfkl

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from bernstein_flow.distributions import BernsteinFlow


class DistributionLambdaTest(tf.test.TestCase):

    def gen_sequential_model(
            self,
            input_shape,
            hidden_units,
            distribution_lambda):
        # Create a trainable distribution using the Sequential API.
        model = tf.keras.models.Sequential([
            tfkl.InputLayer(input_shape=input_shape, dtype=tf.float32),
            # The Dense serves no real purpose; it will change the event_shape.
            tf.keras.layers.Dense(hidden_units, use_bias=False),
            tfp.layers.DistributionLambda(distribution_lambda)
        ])
        return model

    def test_sequential_api(self):
        bernstein_order = 5
        input_shape = (1,)

        normal_model = self.gen_sequential_model(
            input_shape=input_shape,
            hidden_units=2,
            distribution_lambda=lambda pv: tfd.Normal(
                loc=pv[:, 0],
                scale=1e-3 + tf.math.softplus(0.05 * pv[:, 1])))
        trans_model = self.gen_sequential_model(
            input_shape=input_shape,
            hidden_units=4 + bernstein_order,
            distribution_lambda=lambda pv: BernsteinFlow(
                pvector=pv))

        for input_shape in [[1], [2], [32]]:
            normal_dist = normal_model(tf.zeros(input_shape))
            trans_dist = trans_model(tf.zeros(input_shape))

            # Check the distribution.
            self.assertIsInstance(trans_dist, tfd.TransformedDistribution)
            self.assertEqual(normal_dist.batch_shape, trans_dist.batch_shape)
            self.assertEqual(normal_dist.event_shape, trans_dist.event_shape)
            self.assertEqual(normal_dist.sample(5).shape,
                             trans_dist.sample(5).shape)
            self.assertEqual(normal_dist.sample(input_shape).shape,
                             trans_dist.sample(input_shape).shape)
            self.assertEqual(normal_dist.log_prob(tf.zeros((1))).shape,
                             trans_dist.log_prob(tf.zeros((1))).shape)
            self.assertEqual(normal_dist.log_prob(tf.zeros(input_shape)).shape,
                             trans_dist.log_prob(tf.zeros(input_shape)).shape)
            self.assertEqual(normal_dist.log_prob(tf.zeros((10, 1))).shape,
                             trans_dist.log_prob(tf.zeros((10, 1))).shape)
