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
# modified time : 2020-09-11 17:03:42
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-05-15 10:44:23
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from tensorflow_probability import distributions as tfd

from bernstein_flow.bijectors import BernsteinBijector
from bernstein_flow import build_bernstein_flow


class BernsteinFlow():
    """
    This class implements a normalizing flow using Bernstein polynomials.
    """

    def __init__(
            self,
            M: int):
        """
        Constructs a new instance of the flow. It can be used as Distribution a
        distribution with `tfp.layers.DistributionLambda`.

        To use it as a loss function see
        `bernstein_flow.losses.BernsteinFlowLoss`.

        :param      M:    Order of the used Bernstein polynomial bijector.
        :type       M:    int
        """
        self.M = M

    def __call__(self, pvector: tf.Tensor) -> tfd.Distribution:
        """
        Calls `gen_flow(pvector)`.

        :param      pvector:  The paramter vector.
        :type       pvector:  Tensor

        :returns:   The transformed distribution (normalizing flow)
        :rtype:     Distribution
        """

        flow = self.gen_flow(pvector)

        return flow

    def slice_parameter_vectors(self, pvector: tf.Tensor) -> list:
        """
        Returns an unpacked list of parameter vectors.

        :param      pvector:  The parameter vector.
        :type       pvector:  Tensor

        :returns:   unpacked list of parameter vectors.
        :rtype:     list
        """
        p_len = [1, 1, self.M, 1, 1]
        num_dist = pvector.shape[1]
        sliced_pvectors = []
        for d in range(num_dist):
            sliced_pvector = [pvector[:, d, sum(p_len[:i]):(
                sum(p_len[:i + 1]))] for i, p in enumerate(p_len)]
            sliced_pvectors.append(sliced_pvector)
        return sliced_pvectors

    def gen_flow(self, pvector):
        """
        Generate the flow for the given parameter vector. This would be
        typically the output of a neural network.

        :param      pvector:  The paramter vector.
        :type       pvector:  Tensor

        :returns:   The transformed distribution (normalizing flow)
        :rtype:     Distribution
        """
        pvs = self.slice_parameter_vectors(pvector)
        flows = []
        for pv in pvs:
            a1, b1, theta, a2, b2 = pv

            flow = build_bernstein_flow(
                M=self.M,
                a1=tf.math.softplus(a1),
                b1=b1,
                theta=BernsteinBijector.constrain_theta(theta),
                a2=tf.math.softplus(a2),
                b2=b2
            )

            flows.append(flow)
        joint = tfd.JointDistributionSequential(flows, name='joint_bs_flows')
        blkws = tfd.Blockwise(joint)
        return blkws
