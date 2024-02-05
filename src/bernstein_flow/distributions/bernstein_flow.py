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
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)

from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.bijectors import BernsteinBijector


def slice_parameter_vector(pvector: tf.Tensor, p_spec: dict) -> dict:
    """slices parameters of the given size form a tensor.

    :param pvector: The parameter vector.
    :type pvector: tf.Tensor
    :param p_spec: specification of parameter sizes in the form {'parameter_name': size}
    :type p_spec: dict
    :returns: Dictionary containing the sliced parameters
    :rtype: dict

    """
    with tf.name_scope("slice_parameter_vectors"):
        parameters = {}
        offset = 0
        for name, length in p_spec.items():
            # fmt: off
            p = pvector[..., offset:(offset + length)]
            # fmt: on
            offset += length
            parameters[name] = tf.squeeze(p, name=name)
        return parameters


def apply_constraining_bijectors(
    unconstrained_parameters,
    thetas_constrain_fn=None,
):
    """Apply activation functions to raw parameters.

    :param thetas_constrain_fn: Function used to constrain the Bernstein coefficients

    """
    with tf.name_scope("apply_activation"):
        parameters = {}

        for parameter_name, parameter in unconstrained_parameters.items():
            if parameter_name == "thetas" and (thetas_constrain_fn is not None):
                constraining_bijector = thetas_constrain_fn
            else:
                parameter_properties = BernsteinFlow.parameter_properties(
                    dtype=parameter.dtype
                )
                properties = parameter_properties[parameter_name]
                constraining_bijector = properties.default_constraining_bijector_fn()

            parameters[parameter_name] = constraining_bijector(parameter)
        return parameters


def init_bijectors(
    thetas,
    clip_to_bernstein_domain,
    clip_base_distribution,
    a1=None,
    b1=None,
    a2=None,
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
        if tf.is_tensor(b1):
            f1_shift = tfb.Shift(b1, name="shift1")
            bijectors.append(f1_shift)
        if tf.is_tensor(a1):
            f1_scale = tfb.Scale(a1, name="scale1")
            bijectors.append(f1_scale)

        # clip to domain [0, 1]
        if clip_to_bernstein_domain:
            bijectors.append(tfb.Sigmoid(name="sigmoid"))

        # f2: ẑ = Bernstein Polynomial
        f2 = BernsteinBijector(thetas, name="bpoly")
        bijectors.append(f2)

        # clip to range [min(theta), max(theta)]
        if clip_base_distribution:
            bijectors.append(
                tfb.Invert(
                    tfb.SoftClip(
                        high=thetas[..., 0],
                        low=thetas[..., -1],
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


def get_base_distribution(base_distribution, dtype, **kwds):
    if isinstance(base_distribution, tfd.Distribution):
        return base_distribution
    else:
        if base_distribution == "normal":
            default_kwds = dict(loc=tf.convert_to_tensor(0, dtype=dtype), scale=1.0)
            default_kwds.update(**kwds)
            dist = tfd.Normal(**default_kwds)
        elif base_distribution == "truncated_normal":
            default_kwds = dict(
                loc=tf.convert_to_tensor(0, dtype=dtype), scale=1.0, low=-4, high=4
            )
            default_kwds.update(**kwds)
            dist = tfd.TruncatedNormal(**default_kwds)
        elif base_distribution == "log_normal":
            default_kwds = dict(loc=tf.convert_to_tensor(0, dtype=dtype), scale=1.0)
            default_kwds.update(**kwds)
            dist = tfd.LogNormal(**default_kwds)
        elif base_distribution == "logistic":
            default_kwds = dict(loc=tf.convert_to_tensor(0, dtype=dtype), scale=1.0)
            default_kwds.update(**kwds)
            dist = tfd.Logistic(**default_kwds)
        elif base_distribution == "uniform":
            default_kwds = dict(low=tf.convert_to_tensor(0, dtype=dtype), high=1.0)
            default_kwds.update(**kwds)
            dist = tfd.Uniform(**default_kwds)
        elif base_distribution == "kumaraswamy":
            dist = tfd.Kumaraswamy(**kwds)
        else:
            raise ValueError(f"Unsupported distribution type {base_distribution}.")
    return dist


class BernsteinFlow(tfd.TransformedDistribution):
    """
    This class implements a `tfd.TransformedDistribution` using Bernstein
    polynomials as the bijector.
    """

    def __init__(
        self,
        thetas,
        a1=None,
        b1=None,
        a2=None,
        base_distribution=None,
        base_distribution_kwds={},
        clip_to_bernstein_domain=True,
        clip_base_distribution=False,
        name="BernsteinFlow",
        **bb_kwds,
    ) -> tfd.Distribution:
        with tf.name_scope(name) as name:
            parameters = dict(locals())
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

            base_distribution = get_base_distribution(base_distribution, dtype)

            bijector = init_bijectors(
                thetas,
                a1=a1,
                b1=b1,
                a2=a2,
                clip_to_bernstein_domain=clip_to_bernstein_domain,
                clip_base_distribution=clip_base_distribution,
            )

            super().__init__(
                distribution=base_distribution,
                bijector=bijector,
                name=name,
            )

            self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype=None, num_classes=None):
        # Annotations may optionally specify properties, such as `event_ndims`,
        # `default_constraining_bijector_fn`, `specifies_shape`, etc.; see
        # the `ParameterProperties` documentation for details.
        return dict(
            a1=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Softplus(
                    low=dtype_util.eps(dtype)
                )
            ),
            b1=tfp.util.ParameterProperties(),
            thetas=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=get_thetas_constrain_fn, event_ndims=1
            ),
            a2=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Softplus(
                    low=tf.cast(1.0, dtype)
                )
            ),
        )

    @classmethod
    def from_pvector(
        cls,
        pvector,
        scale_data=True,
        shift_data=True,
        scale_base_distribution=True,
        thetas_constrain_fn=None,
        **kwds,
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
                **apply_constraining_bijectors(
                    unconstrained_parameters=slice_parameter_vector(pvector, p_spec),
                    thetas_constrain_fn=thetas_constrain_fn,
                ),
                **kwds,
            )
