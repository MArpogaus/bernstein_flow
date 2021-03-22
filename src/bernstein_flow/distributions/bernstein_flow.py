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

    def __init__(
        self,
        pvector: tf.Tensor,
        scale_base_distribution=True,
        base_distribution=None,
        base_dist_lower_bound=None,
        base_dist_upper_bound=None,
        name="BernsteinFlow",
    ) -> tfd.Distribution:
        """
        Generate the flow for the given parameter vector. This would be
        typically the output of a neural network.

        To use it as a loss function see
        `bernstein_flow.losses.BernsteinFlowLoss`.

        :param      pvector:       The paramter vector.
        :type       pvector:       Tensor

        :returns:   The transformed distribution (normalizing flow)
        :rtype:     Distribution
        """
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([pvector], dtype_hint=tf.float32)

            pvector = tensor_util.convert_nonref_to_tensor(pvector, dtype=dtype)

            shape = prefer_static.shape(pvector)

            if tensorshape_util.rank(pvector.shape) > 1:
                batch_shape = shape[:-1]
            else:
                batch_shape = [1]

            if base_distribution is None:
                base_distribution = tfd.Normal(loc=tf.zeros(batch_shape), scale=1.0)

            tol = 1e-4
            if base_dist_lower_bound is None:
                self.lower_bound = tf.reshape(
                    base_distribution.quantile(tol), tf.concat((batch_shape, [1]), 0)
                )
            else:
                self.lower_bound = base_dist_lower_bound
            if base_dist_upper_bound is None:
                self.upper_bound = tf.reshape(
                    base_distribution.quantile(1 - tol),
                    tf.concat((batch_shape, [1]), 0),
                )
            else:
                self.upper_bound = base_dist_upper_bound

            p_len = [1, 1]
            self.bernstein_order = shape[-1] - 2
            if scale_base_distribution:
                p_len += [1]
                self.bernstein_order -= 1

            p_len.insert(2, self.bernstein_order)

            pv = self.slice_parameter_vectors(pvector, p_len)

            bijector = self.init_bijectors(
                pv,
                scale_base_distribution=scale_base_distribution,
            )

            super().__init__(
                distribution=base_distribution,
                bijector=bijector,
                name=name,
            )

    def slice_parameter_vectors(self, pvector: tf.Tensor, p_len) -> list:
        """
        Returns an unpacked list of parameter vectors.

        :param      pvector:  The parameter vector.
        :type       pvector:  Tensor

        :returns:   unpacked list of parameter vectors.
        :rtype:     list
        """
        sliced_pvector = []
        for i in range(len(p_len)):
            # fmt: off
            p = pvector[..., sum(p_len[:i]):sum(p_len[:i + 1])]
            # fmt: off
            sliced_pvector.append(tf.squeeze(p))

        return sliced_pvector

    def init_bijectors(
        self,
        pv,
        scale_base_distribution,
        name: str = "bernstein_flow",
    ) -> tfb.Bijector:
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

        if scale_base_distribution:
            a1, b1, theta, a2 = pv
        else:
            a1, b1, theta = pv

        def conatrain_dist_scale(scale, low, high):
            scale = (high-low) * tf.math.sigmoid(scale) + low
            return scale

        min_dist_scale = 1.0
        max_dist_scale = 2.0
        if scale_base_distribution:
            scale = conatrain_dist_scale(a2, min_dist_scale, max_dist_scale)[..., None]
        else:
            scale = 1.0

        theta = BernsteinBijector.constrain_theta(
            theta, low=scale * self.lower_bound, high=scale * self.upper_bound
        )

        # f1: ŷ = sigma(a1(x)*y - b1(x))
        f1_scale = tfb.Scale(tf.math.softplus(a1) + 0.1, name="f1_scale")
        bijectors.append(f1_scale)
        f1_shift = tfb.Shift(b1, name="f1_shift")
        bijectors.append(f1_shift)

        # clip to range [0, 1]
        bijectors.append(tfb.Sigmoid())

        # f2: ẑ = Bernstein Polynomial
        f2 = BernsteinBijector(theta=theta, name="f2")
        bijectors.append(f2)

        # f3: z = a2(x)*ẑ - b2(x)
        if scale_base_distribution:
            f3_scale = tfb.Scale(conatrain_dist_scale(a2, min_dist_scale, max_dist_scale), name="f3_scale")
            bijectors.append(f3_scale)

        bijectors = list(reversed(bijectors))

        return tfb.Invert(tfb.Chain(bijectors))

    def _mean(self):
        samples = self.sample(10000)
        return tf.math.reduce_mean(samples, axis=0)
