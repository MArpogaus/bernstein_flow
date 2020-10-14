#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : bernstein_flow.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-05-15 10:44:23
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
# NOTES ######################################################################
#
# This project is following the
# [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-10-14 20:52:24
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-05-15 10:44:23
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from bernstein_flow.bijectors import BernsteinBijector


class BernsteinFlow():
    """
    This class implements a normalizing flow using Bernstein polynomials.
    """

    def __init__(self,
                 order: int):
        """
        Constructs a new instance of the flow. It can be used as Distribution a
        distribution with `tfp.layers.DistributionLambda`.

        To use it as a loss function see
        `bernstein_flow.losses.BernsteinFlowLoss`.

        :param      order:  Order of the used Bernstein polynomial bijector.
        :type       order:  int
        """
        self.order = order

    def __call__(self, pvector: tf.Tensor) -> tfd.Distribution:
        """
        Generate the flow for the given parameter vector. This would be
        typically the output of a neural network.

        :param      pvector:  The paramter vector.
        :type       pvector:  Tensor

        :returns:   The transformed distribution (normalizing flow)
        :rtype:     Distribution
        """
        a1, b1, theta, a2, b2 = self.slice_parameter_vectors(pvector)

        flow = self.parameterize(
            a1=tf.math.softplus(a1),
            b1=b1,
            theta=BernsteinBijector.constrain_theta(theta),
            a2=tf.math.softplus(a2),
            b2=b2
        )

        return flow

    def slice_parameter_vectors(self, pvector: tf.Tensor) -> list:
        """
        Returns an unpacked list of parameter vectors.

        :param      pvector:  The parameter vector.
        :type       pvector:  Tensor

        :returns:   unpacked list of parameter vectors.
        :rtype:     list
        """
        p_len = [1, 1, self.order, 1, 1]

        sliced_pvector = []
        for i in range(len(p_len)):
            p = pvector[..., sum(p_len[:i]): sum(p_len[:i + 1])]
            sliced_pvector.append(tf.squeeze(p))

        a1, b1, theta, a2, b2 = sliced_pvector

        return a1, b1, theta, a2, b2

    def parameterize(self,
                     a1: tf.Tensor,
                     b1: tf.Tensor,
                     theta: tf.Tensor,
                     a2: tf.Tensor,
                     b2: tf.Tensor,
                     name: str = 'bernstein_flow') -> tfd.Distribution:
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
        :rtype:     Distribution
        """
        bijectors = []

        # f1: ŷ = sigma(a1(x)*y - b1(x))
        f1_scale = tfb.Scale(
            a1,
            name=f'{name}_f1_scale'
        )
        bijectors.append(f1_scale)
        f1_shift = tfb.Shift(
            b1,
            name=f'{name}_f1_shift'
        )
        bijectors.append(f1_shift)
        bijectors.append(tfb.Sigmoid())

        # f2: ẑ = Bernstein Polynomial
        f2 = BernsteinBijector(
            order=self.order,
            theta=theta,
            name=f'{name}_f2'
        )
        bijectors.append(f2)

        # f3: z = a2(x)*ẑ - b2(x)
        f3_scale = tfb.Scale(
            a2,
            name=f'{name}_f3_scale'
        )
        bijectors.append(f3_scale)
        f3_shift = tfb.Shift(
            b2,
            name=f'{name}_f3_shift'
        )
        bijectors.append(f3_shift)

        bijectors = list(reversed(bijectors))
        return tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=0., scale=1.),
            bijector=tfb.Invert(tfb.Chain(bijectors)),
            name='NormalTransformedDistribution')
