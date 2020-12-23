#!env python3
# AUTHOR INFORMATION ##########################################################
# file    : bernstein_flow.py
# brief   : [Description]
#
# author  : Marcel Arpogaus
# created : 2020-05-15 10:44:23
# changed : 2020-12-07 12:31:41
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

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from bernstein_flow.bijectors import BernsteinBijector

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


class BernsteinFlow(tfd.TransformedDistribution):
    """
    This class implements a `tfd.TransformedDistribution` using Bernstein
    polynomials as the bijector.
    """

    def __init__(self,
                 pvector: tf.Tensor,
                 distribution: tfd.Distribution = tfd.Normal(loc=0., scale=1.),
                 name='BernsteinFlow'
                 ) -> tfd.Distribution:
        """
        Generate the flow for the given parameter vector. This would be
        typically the output of a neural network.

        To use it as a loss function see
        `bernstein_flow.losses.BernsteinFlowLoss`.

        :param      pvector:       The paramter vector.
        :type       pvector:       Tensor
        :param      distribution:  The base distribution to use.
        :type       distribution:  Distribution

        :returns:   The transformed distribution (normalizing flow)
        :rtype:     Distribution
        """
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([pvector], dtype_hint=tf.float32)

            pvector = tensor_util.convert_nonref_to_tensor(
                pvector, dtype=dtype)

            shape = prefer_static.shape(pvector)
            self.bernstein_order = shape[-1] - 4
            if tensorshape_util.rank(pvector.shape) > 1:
                batch_shape = shape[:-1]
            else:
                batch_shape = [1]

            a1, b1, theta, a2, b2 = self.slice_parameter_vectors(pvector)

            bijector = self.init_bijectors(
                a1=tf.math.softplus(a1),
                b1=b1,
                theta=BernsteinBijector.constrain_theta(theta),
                a2=tf.math.softplus(a2),
                b2=b2
            )

            super().__init__(
                distribution=tfd.Normal(loc=0., scale=1.),
                bijector=bijector,
                batch_shape=batch_shape,
                name=name)

    def slice_parameter_vectors(self, pvector: tf.Tensor) -> list:
        """
        Returns an unpacked list of parameter vectors.

        :param      pvector:  The parameter vector.
        :type       pvector:  Tensor

        :returns:   unpacked list of parameter vectors.
        :rtype:     list
        """
        p_len = [1, 1, self.bernstein_order, 1, 1]

        sliced_pvector = []
        for i in range(len(p_len)):
            p = pvector[..., sum(p_len[:i]):sum(p_len[:i + 1])]
            sliced_pvector.append(tf.squeeze(p))

        a1, b1, theta, a2, b2 = sliced_pvector

        return a1, b1, theta, a2, b2

    def init_bijectors(self,
                       a1: tf.Tensor,
                       b1: tf.Tensor,
                       theta: tf.Tensor,
                       a2: tf.Tensor,
                       b2: tf.Tensor,
                       name: str = 'bernstein_flow') -> tfb.Bijector:
        """
        Builds a normalizing flow using a Bernstein polynomial as Bijector.

        :param      a1:     The scale of f1.
        :type       a1:     Tensor
        :param      b1:     The shift of f1.
        :type       b1:     Tensor
        :param      theta:  The Bernstein coefficients.
        :type       theta:  Tensor
        :param      a2:     The scale of f3.
        :type       a2:     Tensor
        :param      b2:     The shift of f3.
        :type       b2:     Tensor
        :param      name:   The name to give Ops created by the initializer.
        :type       name:   string

        :returns:   The Bernstein flow.
        :rtype:     Bijector
        """
        bijectors = []

        # f1: ŷ = sigma(a1(x)*y - b1(x))
        f1_scale = tfb.Scale(
            a1,
            name='f1_scale'
        )
        bijectors.append(f1_scale)
        f1_shift = tfb.Shift(
            b1,
            name='f1_shift'
        )
        bijectors.append(f1_shift)

        # clip to range [0, 1]
        bijectors.append(
            tfb.SoftClip(
                low=0,
                high=1,
                hinge_softness=1.5
            )
        )

        # f2: ẑ = Bernstein Polynomial
        f2 = BernsteinBijector(
            theta=theta,
            name='f2'
        )
        bijectors.append(f2)

        # clip to range [min(theta), max(theta)]
        # bijectors.append(
        #     tfb.Invert(
        #         tfb.SoftClip(
        #             high=tf.math.reduce_max(theta, axis=-1),
        #             low=tf.math.reduce_min(theta, axis=-1),
        #             hinge_softness=0.5
        #         )
        #     )
        # )
        # f3: z = a2(x)*ẑ - b2(x)
        f3_scale = tfb.Scale(
            a2,
            name='f3_scale'
        )
        bijectors.append(f3_scale)
        f3_shift = tfb.Shift(
            b2,
            name='f3_shift'
        )
        bijectors.append(f3_shift)

        bijectors = list(reversed(bijectors))

        return tfb.Invert(tfb.Chain(bijectors))

    def _mean(self):
        samples = self.sample(10000)
        return tf.math.reduce_mean(samples, axis=0)
