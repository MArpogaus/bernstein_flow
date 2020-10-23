#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : test_bernstein_bijector.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-10-16 08:12:04
# changed : 2020-10-23 18:21:07
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
import numpy as np
import tensorflow as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from bernstein_flow.bijectors import BernsteinBijector


class BernsteinBijectorTest(tf.test.TestCase):

    def test_unbatched(self):
        order = 9
        theta = BernsteinBijector.constrain_theta(
            np.random.normal(0, 2, order).astype(np.float32)
        )

        bb = BernsteinBijector(
            order=order,
            theta=theta
        )

        x = np.linspace(0, 1, 10, dtype=np.float32)

        forward_x = bb.forward(x)
        # Use identity to invalidate cache.
        inverse_y = bb.inverse(tf.identity(forward_x))
        forward_inverse_y = bb.forward(inverse_y)

        fldj = bb.forward_log_det_jacobian(x, event_ndims=1)
        # Use identity to invalidate cache.
        ildj = bb.inverse_log_det_jacobian(
            tf.identity(forward_x), event_ndims=1)

        print(forward_x, inverse_y)

        self.assertAllClose(x, inverse_y, rtol=1e-6, atol=1e-5)
        self.assertAllClose(forward_x, forward_inverse_y, rtol=1e-6, atol=1e-5)
        self.assertAllClose(ildj, -fldj, rtol=1e-6, atol=0.)

    def test_batch(self, batch_size=5):
        order = 9

        theta = BernsteinBijector.constrain_theta(
            np.random.normal(0, 2, (batch_size, order)).astype(np.float32)
        )

        bb = BernsteinBijector(
            order=order,
            theta=theta
        )

        x = np.linspace(0, 1, 10, dtype=np.float32)[..., None]

        forward_x = bb.forward(x)

        # Use identity to invalidate cache.
        inverse_y = bb.inverse(tf.identity(forward_x))
        forward_inverse_y = bb.forward(inverse_y)

        fldj = bb.forward_log_det_jacobian(x, event_ndims=1)
        # Use identity to invalidate cache.
        ildj = bb.inverse_log_det_jacobian(
            tf.identity(forward_x), event_ndims=1)

        self.assertAllClose(forward_x, forward_inverse_y, rtol=1e-3, atol=1e-2)
        self.assertAllClose(x, inverse_y, rtol=1e-6, atol=1e-5)
        self.assertAllClose(ildj, -fldj, rtol=1e-6, atol=0.)
