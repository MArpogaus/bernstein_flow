# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : bernstein_flow.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-07-12 14:55:22 (Marcel Arpogaus)
# changed : 2024-07-12 15:21:35 (Marcel Arpogaus)

# %% License ###################################################################
# Copyright 2024 Marcel Arpogaus
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

# %% Description ###############################################################
"""Normalizing flow using Bernstein Polynomial as transformation function."""

# %% imports ###################################################################
from typing import Any, Callable, Dict, Optional

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
from bernstein_flow.bijectors import BernsteinPolynomial


# %% functions #################################################################
def slice_parameter_vector(
    params: tf.Tensor, p_spec: Dict[str, int]
) -> Dict[str, tf.Tensor]:
    """Slice parameters of the given size from a tensor.

    Parameters
    ----------
    params
        The parameter vector.
    p_spec
        Specification of parameter sizes in the form {'parameter_name': size}.

    Returns
    -------
    Dict[str, tf.Tensor]
        Dictionary containing the sliced parameters.

    """
    with tf.name_scope("slice_parameter_vectors"):
        parameters = {}
        offset = 0
        for name, length in p_spec.items():
            p = params[..., offset : (offset + length)]
            offset += length
            parameters[name] = tf.squeeze(p, name=name)
        return parameters


def apply_constraining_bijectors(
    unconstrained_parameters: Dict[str, tf.Tensor],
    thetas_constraint_fn: Optional[Callable] = None,
) -> Dict[str, tf.Tensor]:
    """Apply activation functions to raw parameters.

    Parameters
    ----------
    unconstrained_parameters
        Dictionary of raw parameters.
    thetas_constraint_fn
        Function used to constrain the Bernstein coefficients, by default None.

    Returns
    -------
    Dict[str, tf.Tensor]
        Dictionary with constrained parameters.

    """
    with tf.name_scope("apply_activation"):
        parameters = {}

        for parameter_name, parameter in unconstrained_parameters.items():
            if parameter_name == "thetas" and (thetas_constraint_fn is not None):
                constraining_bijector = thetas_constraint_fn
            else:
                parameter_properties = BernsteinFlow.parameter_properties(
                    dtype=parameter.dtype
                )
                properties = parameter_properties[parameter_name]
                constraining_bijector = properties.default_constraining_bijector_fn()

            parameters[parameter_name] = constraining_bijector(parameter)
        return parameters


def init_bijectors(
    thetas: tf.Tensor,
    clip_to_bernstein_domain: bool,
    a1: Optional[tf.Tensor] = None,
    b1: Optional[tf.Tensor] = None,
    a2: Optional[tf.Tensor] = None,
    **bernstein_bijector_kwargs: Dict[str, Any],
) -> tfb.Bijector:
    """Build a normalizing flow using a Bernstein polynomial as Bijector.

    Parameters
    ----------
    thetas
        The Bernstein coefficients.
    clip_to_bernstein_domain
        Whether to clip to the Bernstein domain [0, 1].
    a1
        The scale of f1., by default None.
    b1
        The shift of f1., by default None.
    a2
        The scale of f3., by default None.
    bernstein_bijector_kwargs
        Keyword arguments passed to the `BernsteinPolynomial`

    Returns
    -------
    tfb.Bijector
        The Bernstein flow.

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
        f2 = BernsteinPolynomial(
            thetas, name="bernstein_bijector", **bernstein_bijector_kwargs
        )
        bijectors.append(f2)

        # f3: z = a2(x)*ẑ - b2(x)
        if tf.is_tensor(a2):
            f3_scale = tfb.Scale(a2, name="scale2")
            bijectors.append(f3_scale)

        bijectors = list(reversed(bijectors))

        return tfb.Invert(tfb.Chain(bijectors))


def get_base_distribution(
    base_distribution: str, dtype: tf.DType, **kwargs: Dict[str, Any]
) -> tfd.Distribution:
    """Get an instance of a base distribution.

    Parameters
    ----------
    base_distribution
        Name of the base distribution.
    dtype
        Data type of the distribution.
    kwargs
        Keyword arguments passed to the Distribution class.

    Returns
    -------
    tfd.Distribution
        Instance of the base distribution.

    Raises
    ------
    ValueError
        If an unknown distribution name is given.

    """
    if isinstance(base_distribution, tfd.Distribution):
        return base_distribution
    else:
        if base_distribution == "normal":
            default_kwargs = dict(loc=tf.cast(0, dtype=dtype), scale=1.0)
            default_kwargs.update(**kwargs)
            dist = tfd.Normal(**default_kwargs)
        elif base_distribution == "truncated_normal":
            default_kwargs = dict(
                loc=tf.cast(0, dtype=dtype), scale=1.0, low=-4, high=4
            )
            default_kwargs.update(**kwargs)
            dist = tfd.TruncatedNormal(**default_kwargs)
        elif base_distribution == "log_normal":
            default_kwargs = dict(loc=tf.cast(0, dtype=dtype), scale=1.0)
            default_kwargs.update(**kwargs)
            dist = tfd.LogNormal(**default_kwargs)
        elif base_distribution == "logistic":
            default_kwargs = dict(loc=tf.cast(0, dtype=dtype), scale=1.0)
            default_kwargs.update(**kwargs)
            dist = tfd.Logistic(**default_kwargs)
        elif base_distribution == "uniform":
            default_kwargs = dict(low=tf.cast(0, dtype=dtype), high=1.0)
            default_kwargs.update(**kwargs)
            dist = tfd.Uniform(**default_kwargs)
        elif base_distribution == "kumaraswamy":
            dist = tfd.Kumaraswamy(**kwargs)
        else:
            raise ValueError(f"Unsupported distribution type {base_distribution}.")
    return dist


# %% classes ###################################################################
class BernsteinFlow(tfd.TransformedDistribution):
    """Implement a `tfd.TransformedDistribution` using Bernstein polynomials."""

    def __init__(
        self,
        thetas: tf.Tensor,
        a1: Optional[tf.Tensor] = None,
        b1: Optional[tf.Tensor] = None,
        a2: Optional[tf.Tensor] = None,
        base_distribution: str = "normal",
        base_distribution_kwargs: Dict[str, Any] = {},
        clip_to_bernstein_domain: bool = False,
        name: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the BernsteinFlow.

        Parameters
        ----------
        thetas
            The Bernstein coefficients.
        a1
            The scale of f1., by default None.
        b1
            The shift of f1., by default None.
        a2
            The scale of f3., by default None.
        base_distribution
            The base distribution, by default "normal".
        base_distribution_kwargs
            Keyword arguments of the base distribution, by default {}.
        clip_to_bernstein_domain
            Whether to clip to the Bernstein domain [0, 1], by default False.
        name
            The name of the flow, by default "BernsteinFlow".
        kwargs
            Keyword arguments passed to `init_bijectors`.

        """
        parameters = dict(locals())
        with tf.name_scope(name or "BernsteinFlow") as name:
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

            base_distribution = get_base_distribution(
                base_distribution, dtype, **base_distribution_kwargs
            )

            bijector = init_bijectors(
                thetas,
                a1=a1,
                b1=b1,
                a2=a2,
                clip_to_bernstein_domain=clip_to_bernstein_domain,
                **kwargs,
            )

            super().__init__(
                distribution=base_distribution,
                bijector=bijector,
                name=name,
                parameters=parameters,
            )

    @classmethod
    def _parameter_properties(cls, dtype=None, num_classes=None):
        return dict(
            a1=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Softplus(
                    low=dtype_util.eps(dtype)
                )
            ),
            b1=tfb.Shift.parameter_properties(dtype)["shift"],
            thetas=BernsteinPolynomial.parameter_properties(dtype)["thetas"],
            a2=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=lambda: tfb.Softplus(
                    low=tf.cast(1.0, dtype)
                )
            ),
        )

    @staticmethod
    def new(
        params: tf.Tensor,
        scale_data: bool,
        shift_data: bool,
        scale_base_distribution: bool,
        get_thetas_constraint_fn: Callable = get_thetas_constrain_fn,
        base_distribution: str = "normal",
        base_distribution_kwargs: Dict[str, Any] = {},
        clip_to_bernstein_domain: bool = False,
        name: Optional[str] = None,
        bernstein_bijector_kwargs: Dict[str, Any] = {},
        **kwargs: Dict[str, Any],
    ) -> "BernsteinFlow":
        """Create the distribution instance from a `params` vector.

        Parameters
        ----------
        params
            The parameters of the flow.
        scale_data
            Whether to scale the data.
        shift_data
            Whether to shift the data.
        scale_base_distribution
            Whether to scale the base distribution.
        get_thetas_constraint_fn
            Function returning a constrain function for the Bernstein coefficients.
        base_distribution
            The base distribution, by default "normal".
        base_distribution_kwargs
            Keyword arguments for the base distribution, by default {}.
        clip_to_bernstein_domain
            Whether to clip to the Bernstein domain [0, 1], by default False.
        name
            The name of the flow, by default "BernsteinFlow".
        bernstein_bijector_kwargs
            Keyword arguments for the Bernstein bijector, by default {}.
        kwargs
            Keyword arguments passed to `get_thetas_constraint_fn`.

        Returns
        -------
        BernsteinFlow
            The resulting flow.

        """
        with tf.name_scope(name or "BernsteinFlow"):
            dtype = dtype_util.common_dtype([params], dtype_hint=tf.float32)

            params = tensor_util.convert_nonref_to_tensor(
                params, dtype=dtype, name="pvector"
            )

            shape = prefer_static.shape(params)

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
            return BernsteinFlow(
                **apply_constraining_bijectors(
                    unconstrained_parameters=slice_parameter_vector(params, p_spec),
                    thetas_constraint_fn=get_thetas_constraint_fn(**kwargs),
                ),
                base_distribution=base_distribution,
                base_distribution_kwargs=base_distribution_kwargs,
                clip_to_bernstein_domain=clip_to_bernstein_domain,
                name=name,
                **bernstein_bijector_kwargs,
            )
