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
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)

from bernstein_flow.bijectors import BernsteinBijector
from bernstein_flow.bijectors.bernstein import constrain_thetas


def slice_parameter_vector(pvector: tf.Tensor, p_spec: dict = None) -> dict:
    """slices parameters of the given size form a tensor.

    :param pvector: The parameter vector.
    :type pvector: tf.Tensor
    :param p_spec: specification of parameter sizes in the form {'parameter_name': size}
    :type p_spec: dict
    :returns: Dictionary containing the scliced parameters
    :rtype: dict

    """
    with tf.name_scope("slice_parameter_vectors"):
        if p_spec is None:
            shape = prefer_static.shape(pvector)
            bernstein_order = shape[-1] - 3
            p_spec = {
                "a1": 1,
                "b1": 1,
                "thetas": bernstein_order,
                "a2": 1,
            }

        parameters = {}
        offset = 0
        for name, length in p_spec.items():
            # fmt: off
            p = pvector[..., offset:(offset + length)]
            # fmt: on
            offset += length
            parameters[name] = tf.squeeze(p, name=name)
        return parameters


def ensure_positive(x: tf.Tensor, min_value: float = 1e-2, name: str = None):
    """Activation function wich ensures that all given values are positive <= min_value.

    :param x: Tensor to evaluate the function on.
    :param min_value: minimum value (optional)
      Default Value: 1e-8
    :param name: name for the operation (optional)
    :type name: str
    :returns: Tensor with positive values

    """
    with tf.name_scope("ensure_positive"):
        scale = tf.math.softplus(x) + min_value
        return tf.identity(scale, name=name)


def apply_activation(
    thetas, a1=None, b1=None, a2=None, support=None, allow_values_outside_support=False
):
    with tf.name_scope("apply_activation"):
        if support is None:
            support = (tf.constant(-4.0, name="low"), tf.constant(4.0, name="high"))
        low, high = support
        result = {}
        if tf.is_tensor(a1):
            result["a1"] = ensure_positive(a1, name="a1")
        if tf.is_tensor(b1):
            result["b1"] = tf.identity(b1, name="b1")
        if tf.is_tensor(a2):
            result["a2"] = ensure_positive(a2, min_value=1.0, name="a2")
            low *= result["a2"][..., tf.newaxis]
            high *= result["a2"][..., tf.newaxis]

        result["thetas"] = constrain_thetas(
            thetas_unconstrained=thetas,
            low=low,
            high=high,
            allow_values_outside_support=allow_values_outside_support,
        )

        return result


def init_bijectors(
    thetas,
    a1=None,
    b1=None,
    a2=None,
    clip_to_bernstein_domain=True,
    clip_base_distribution=False,
    bb_class=BernsteinBijector,
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
    with tf.name_scope("init_bijectors"):
        bijectors = []

        # f1: ŷ = sigma(a1(x)*y - b1(x))
        if tf.is_tensor(a1):
            f1_scale = tfb.Scale(a1, name="scale1")
            bijectors.append(f1_scale)
        if tf.is_tensor(b1):
            f1_shift = tfb.Shift(b1, name="shift1")
            bijectors.append(f1_shift)

        # clip to domain [0, 1]
        if clip_to_bernstein_domain:
            bijectors.append(tfb.Sigmoid(name="sigmoid"))

        # f2: ẑ = Bernstein Polynomial
        f2 = bb_class(thetas, name="bpoly")
        bijectors.append(f2)

        # clip to range [min(theta), max(theta)]
        if clip_base_distribution:
            bijectors.append(
                tfb.Invert(
                    tfb.SoftClip(
                        high=tf.math.reduce_max(thetas, axis=-1),
                        low=tf.math.reduce_min(thetas, axis=-1),
                        hinge_softness=0.5,
                        name="soft_clip_base_distribution",
                    )
                )
            )

        # f3: z = a2(x)*ẑ - b2(x)
        if tf.is_tensor(a2):
            f3_scale = tfb.Scale(a2, name="scale2")
            bijectors.append(f3_scale)

        bijectors = list(reversed(bijectors))

        return tfb.Invert(tfb.Chain(bijectors))


class BernsteinFlow(tfd.TransformedDistribution):
    """
    This class implements a `tfd.TransformedDistribution` using Bernstein
    polynomials as the bijector.
    """

    def __init__(
        self,
        thetas: tf.Tensor,
        a1=None,
        b1=None,
        a2=None,
        base_distribution=None,
        clip_to_bernstein_domain=True,
        clip_base_distribution=False,
        bb_class=BernsteinBijector,
        name="BernsteinFlow",
    ) -> tfd.Distribution:
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([thetas, a1, b1, a2], dtype_hint=tf.float32)

            thetas = tensor_util.convert_nonref_to_tensor(
                thetas, dtype=dtype, name="thetas"
            )

            if tf.is_tensor(a1):
                a1 = tensor_util.convert_nonref_to_tensor(a1, dtype=dtype, name="a1")

            if tf.is_tensor(b1):
                b1 = tensor_util.convert_nonref_to_tensor(b1, dtype=dtype, name="b1")

            if tf.is_tensor(a2):
                a2 = tensor_util.convert_nonref_to_tensor(a2, dtype=dtype, name="a2")

            shape = prefer_static.shape(thetas)

            if base_distribution is None:
                base_distribution = tfd.Normal(loc=tf.zeros(shape[:-1]), scale=1.0)

            bijector = init_bijectors(
                thetas,
                a1=a1,
                b1=b1,
                a2=a2,
                clip_to_bernstein_domain=clip_to_bernstein_domain,
                clip_base_distribution=clip_base_distribution,
                bb_class=bb_class,
            )

            super().__init__(
                distribution=base_distribution, bijector=bijector, name=name,
            )

    @classmethod
    def from_pvector(
        cls,
        pvector,
        scale_data=True,
        shift_data=True,
        scale_base_distribution=True,
        support=None,
        allow_values_outside_support=False,
        **kwds
    ):
        with tf.name_scope("from_pvector"):
            dtype = dtype_util.common_dtype([pvector], dtype_hint=tf.float32)

            pvector = tensor_util.convert_nonref_to_tensor(
                pvector, dtype=dtype, name="pvector"
            )

            shape = prefer_static.shape(pvector)

            p_spec = {}

            bernstein_order = shape[-1]

            if scale_data:
                p_spec["a1"] = 1
                bernstein_order -= 1

            if shift_data:
                p_spec["b1"] = 1
                bernstein_order -= 1

            if scale_base_distribution:
                p_spec["a2"] = 1
                bernstein_order -= 1

            p_spec["thetas"] = bernstein_order
            return cls(
                **apply_activation(
                    **slice_parameter_vector(pvector, p_spec),
                    support=support,
                    allow_values_outside_support=allow_values_outside_support
                ),
                **kwds
            )
