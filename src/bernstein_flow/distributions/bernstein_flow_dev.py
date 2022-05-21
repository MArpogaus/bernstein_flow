import tensorflow as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

# from bernstein_flow.bijectors import BernsteinBijector

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


class BernsteinFlowScale(tfd.TransformedDistribution):
    """
    This class implements a `tfd.TransformedDistribution` using Bernstein
    polynomials as the bijector.
    """

    def __init__(
        self,
        pvector: tf.Tensor,
        scale_base_distribution=True,
        clip_to_support=True,
        scale_data=True,
        shift_data=True,
        base_distribution=None,
        base_dist_lower_bound=None,
        base_dist_upper_bound=None,
        bb_class=BernsteinBijectorScale,
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

            pvector = tensor_util.convert_nonref_to_tensor(
                pvector, dtype=dtype, name="pvector"
            )

            shape = prefer_static.shape(pvector)

            if tensorshape_util.rank(pvector.shape) > 1:
                batch_shape = shape[:-1]
            else:
                batch_shape = [1]

            if base_distribution is None:
                base_distribution = tfd.Normal(loc=tf.zeros(batch_shape), scale=1.0)

            with tf.name_scope("support"):
                tol = 1e-3
                if base_dist_lower_bound is None:
                    self.lower_bound = tf.reshape(
                        base_distribution.quantile(tol),
                        tf.concat((batch_shape, [1]), 0),
                        name="lower_bound",
                    )
                else:
                    self.lower_bound = base_dist_lower_bound
                if base_dist_upper_bound is None:
                    self.upper_bound = tf.reshape(
                        base_distribution.quantile(1 - tol),
                        tf.concat((batch_shape, [1]), 0),
                        name="upper_bound",
                    )
                else:
                    self.upper_bound = base_dist_upper_bound

                self.lower_bound = tf.identity(self.lower_bound, name="lower_bound")
                self.upper_bound = tf.identity(self.upper_bound, name="upper_bound")

            p_spec = {}
            self.bernstein_order = shape[-1]

            if scale_data:
                p_spec["a1"] = 1
                self.bernstein_order -= 1

            if shift_data:
                p_spec["b1"] = 1
                self.bernstein_order -= 1

            if scale_base_distribution:
                p_spec["a2"] = 1
                self.bernstein_order -= 1

            p_spec["theta"] = self.bernstein_order

            parameters = self.slice_parameter_vectors(pvector, p_spec)

            bijector = self.init_bijectors(
                **parameters, clip_to_support=clip_to_support, bb_class=bb_class
            )

            super().__init__(
                distribution=base_distribution,
                bijector=bijector,
                name=name,
            )

    def slice_parameter_vectors(self, pvector: tf.Tensor, p_spec: dict) -> list:
        """
        Returns an unpacked list of parameter vectors.

        :param      pvector:  The parameter vector.
        :type       pvector:  Tensor

        :returns:   unpacked list of parameter vectors.
        :rtype:     list
        """
        with tf.name_scope("slice_parameter_vectors"):
            parameters = {}
            offset = 0
            for name, length in p_spec.items():
                p = pvector[..., offset : (offset + length)]
                offset += length
                parameters[name] = tf.squeeze(p, name=name)
            return parameters

    def init_bijectors(
        self,
        theta,
        a1=None,
        b1=None,
        a2=None,
        clip_to_support=True,
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

            def conatrain_dist_scale(scale, low, high):
                scale = (high - low) * tf.math.sigmoid(scale) + low
                return scale

            def ensure_positive(scale, min_scale=1e-8):
                with tf.name_scope("ensure_positive"):
                    scale = tf.math.softplus(scale) + min_scale
                    return scale

            if tf.is_tensor(a2):
                scale = ensure_positive(a2, 1.0)[..., None]
            else:
                scale = 1.0

            theta = bb_class.constrain_theta(
                theta, low=scale * self.lower_bound, high=scale * self.upper_bound
            )

            # f1: ŷ = sigma(a1(x)*y - b1(x))
            if tf.is_tensor(a1):
                f1_scale = tfb.Scale(ensure_positive(a1), name="f1_scale")
                bijectors.append(f1_scale)
            if tf.is_tensor(b1):
                f1_shift = tfb.Shift(b1, name="f1_shift")
                bijectors.append(f1_shift)

            # clip to range [0, 1]
            if clip_to_support:
                bijectors.append(tfb.Sigmoid())

            # f2: ẑ = Bernstein Polynomial
            f2 = bb_class(theta=theta, name="f2")
            bijectors.append(f2)

            # f3: z = a2(x)*ẑ - b2(x)
            if tf.is_tensor(a2):
                f3_scale = tfb.Scale(ensure_positive(a2, 1.0), name="f3_scale")
                bijectors.append(f3_scale)

            bijectors = list(reversed(bijectors))

            return tfb.Invert(tfb.Chain(bijectors))

    def _mean(self):
        samples = self.sample(10000)
        return tf.math.reduce_mean(samples, axis=0)
