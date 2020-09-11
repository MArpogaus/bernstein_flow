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
# modified time : 2020-09-11 15:19:56
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-05-15 10:44:23
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from ..bijectors import BernsteinBijector


def bernstein_flow(M, a1, b1, theta, a2, b2, name='bsf'):
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

    # f2: ẑ = Berstein
    f2 = BernsteinBijector(
        len_theta=M,
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
        event_shape=[1],
        name='NormalTransformedDistribution')


class BernsteinFlow():
    def __init__(
            self,
            M):
        self.M = M

    def __call__(self, pvector):

        flow = self.gen_flow(pvector)

        return flow

    def slice_parameter_vectors(self, pvector):
        """ Returns an unpacked list of paramter vectors.
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
        pvs = self.slice_parameter_vectors(pvector)
        flows = []
        for pv in pvs:
            a1, b1, theta, a2, b2 = pv

            flow = bernstein_flow(
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
