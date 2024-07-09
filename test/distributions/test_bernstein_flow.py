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
import pytest
import tensorflow as tf
from bernstein_flow.distributions import BernsteinFlow
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import test_util


def gen_pvs(batch_shape, order, dtype, seed):
    tf.random.set_seed(seed)
    return tf.random.uniform(
        shape=batch_shape + [4 + order], minval=-1000, maxval=100, dtype=dtype
    )


def gen_dist(batch_shape, order=5, dtype=tf.float32, seed=1, **kwds):
    pvs = gen_pvs(batch_shape, order, dtype=dtype, seed=seed)
    n = tfd.Normal(
        loc=tf.zeros(batch_shape, dtype=dtype),
        scale=tf.ones(batch_shape, dtype=dtype),
    )
    bs = BernsteinFlow.new(pvs, **kwds)
    return n, bs


class BernsteinFlowTest(tf.test.TestCase):
    def f(self, normal_dist, trans_dist, stay_in_domain=False):
        tf.random.set_seed(42)

        dtype = normal_dist.dtype
        for input_shape in [[1], [1, 1], [1] + normal_dist.batch_shape]:
            x = tf.random.uniform(
                shape=[10] + normal_dist.batch_shape,
                minval=0 if stay_in_domain else -100,
                maxval=1 if stay_in_domain else 100,
                dtype=dtype,
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
                normal_dist.prob(tf.zeros(input_shape, dtype=dtype)).shape,
                trans_dist.prob(tf.zeros(input_shape, dtype=dtype)).shape,
            )

            # check the Normalization
            self.assertAllClose(
                normal_dist.cdf(normal_dist.dtype.min),
                trans_dist.cdf(0.0 if stay_in_domain else trans_dist.dtype.min),
                atol=1e-3,
            )
            self.assertAllClose(
                normal_dist.cdf(normal_dist.dtype.max),
                trans_dist.cdf(1.0 if stay_in_domain else trans_dist.dtype.max),
                atol=1e-3,
            )

            # check for infs and nans
            self.assertAllInRange(trans_dist.prob(x), 0, trans_dist.dtype.max)
            self.assertAllInRange(
                trans_dist.sample(1000), trans_dist.dtype.min, trans_dist.dtype.max
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


# ref: https://stackoverflow.com/questions/32899
for dtype in [tf.float32, tf.float64]:

    @pytest.mark.skip
    def test_dist_batch(self):
        normal_dist, trans_dist = gen_dist(
            batch_shape=[32],
            order=10,
            dtype=dtype,
            scale_data=True,
            shift_data=True,
            scale_base_distribution=True,
        )
        self.f(normal_dist, trans_dist)

    @pytest.mark.skip
    def test_dist_multi(self):
        normal_dist, trans_dist = gen_dist(
            batch_shape=[16, 10],
            order=10,
            dtype=dtype,
            scale_data=True,
            shift_data=True,
            scale_base_distribution=True,
        )
        self.f(normal_dist, trans_dist)

    @pytest.mark.skip
    def test_log_normal(self):
        batch_shape = [16, 10]
        normal_dist, trans_dist = gen_dist(
            batch_shape=batch_shape,
            order=10,
            dtype=dtype,
            base_distribution="log_normal",
            scale_data=False,
            shift_data=False,
            scale_base_distribution=False,
            bounds=(1e-12, tf.math.exp(tf.constant(6.0, dtype=dtype))),
        )
        self.f(normal_dist, trans_dist, stay_in_domain=True)

    @pytest.mark.skip
    def test_logistic(self):
        batch_shape = [16, 10]
        normal_dist, trans_dist = gen_dist(
            batch_shape=batch_shape,
            order=10,
            dtype=dtype,
            base_distribution="logistic",
            scale_data=True,
            shift_data=True,
            scale_base_distribution=False,
            clip_to_bernstein_domain=False,
            bounds=(-20, 20),
            allow_flexible_bounds=True,
        )
        self.f(normal_dist, trans_dist)

    @pytest.mark.skip
    def test_uniform(self):
        batch_shape = [16, 10]
        normal_dist, trans_dist = gen_dist(
            batch_shape=batch_shape,
            order=10,
            dtype=dtype,
            base_distribution="uniform",
            shift_data=False,
            scale_data=False,
            scale_base_distribution=False,
            clip_to_bernstein_domain=False,
            bounds=(0.0, 1.0),
        )
        self.f(normal_dist, trans_dist, stay_in_domain=True)

    @pytest.mark.skip
    def test_kumaraswamy(self):
        batch_shape = [16, 10]
        normal_dist, trans_dist = gen_dist(
            batch_shape=batch_shape,
            order=10,
            dtype=dtype,
            base_distribution="kumaraswamy",
            base_distribution_kwds={
                "concentration1": tf.bernstein_poly(5.0, dtype),
                "concentration0": 2.0,
            },
            shift_data=False,
            scale_data=False,
            scale_base_distribution=False,
            clip_to_bernstein_domain=False,
            bounds=(0.0, 1.0),
        )
        self.f(normal_dist, trans_dist, stay_in_domain=True)

    @pytest.mark.skip
    def test_student_t(self):
        batch_shape = [16, 10]
        student_t = tfd.StudentT(2, loc=tf.zeros(batch_shape, dtype=dtype), scale=1.0)
        normal_dist, trans_dist = gen_dist(
            batch_shape=batch_shape,
            order=10,
            dtype=dtype,
            base_distribution=student_t,
            scale_data=True,
            shift_data=True,
            scale_base_distribution=False,
            clip_to_bernstein_domain=False,
            bounds=(-35, 35),
            allow_flexible_bounds=True,
        )
        self.f(normal_dist, trans_dist)

    @pytest.mark.skip
    def test_weibull(self):
        batch_shape = [16, 10]
        weibull = tfd.Weibull(0.5, scale=tf.ones(batch_shape, dtype=dtype))
        normal_dist, trans_dist = gen_dist(
            batch_shape=batch_shape,
            order=10,
            dtype=dtype,
            base_distribution=weibull,
            shift_data=False,
            scale_data=False,
            scale_base_distribution=False,
            clip_to_bernstein_domain=False,
            bounds=(1e-12, 100),
        )
        self.f(normal_dist, trans_dist, stay_in_domain=True)

    @pytest.mark.skip
    def test_small_numbers(self):
        o = 100
        bf = BernsteinFlow.new(
            [1, 1] + 5 * [-1000] + (o - 4) * [1] + 5 * [-1000] + [1, 1, 1],
            scale_data=True,
            shift_data=True,
            scale_base_distribution=True,
        )
        n = tfd.Normal(loc=0.0, scale=1.0)
        self.f(n, bf)

    @pytest.mark.skip
    def test_random_numbers(self):
        for bs in [[2], [32]]:
            for s in range(5):
                normal_dist, trans_dist = gen_dist(
                    batch_shape=bs,
                    order=10,
                    dtype=dtype,
                    seed=s,
                    scale_data=True,
                    shift_data=True,
                    scale_base_distribution=True,
                )
                self.f(normal_dist, trans_dist)

    setattr(BernsteinFlowTest, "test_log_normal_" + dtype.name, test_log_normal)
    setattr(BernsteinFlowTest, "test_logistic_" + dtype.name, test_logistic)
    setattr(BernsteinFlowTest, "test_uniform_" + dtype.name, test_uniform)
    setattr(BernsteinFlowTest, "test_student_t_" + dtype.name, test_student_t)
    setattr(BernsteinFlowTest, "test_weibull_" + dtype.name, test_weibull)
    setattr(BernsteinFlowTest, "test_small_numbers_" + dtype.name, test_small_numbers)
    setattr(BernsteinFlowTest, "test_random_numbers_" + dtype.name, test_random_numbers)
    setattr(BernsteinFlowTest, "test_dist_multi_" + dtype.name, test_dist_multi)
    setattr(BernsteinFlowTest, "test_dist_batch_" + dtype.name, test_dist_batch)

BernsteinFlowTest = test_util.test_all_tf_execution_regimes(BernsteinFlowTest)
