import tensorflow as tf
from . import BernsteinBijector


class BernsteinBijectorLinearExtrapolate(BernsteinBijector):
    def __init__(self, *args, **kwds):
        super().__init__(*args, clip_inverse=0.0, **kwds)
        # save slope on boundaries for interpolation
        self.z_min = tf.math.reduce_min(self.thetas, axis=-1)
        self.z_max = tf.math.reduce_max(self.thetas, axis=-1)

        a = tf.cast(self.order - 1, self.thetas.dtype)
        self.a0 = a * (self.thetas[..., 1] - self.thetas[..., 0])
        self.b0 = self.z_min

        self.a1 = a * (self.thetas[..., -1] - self.thetas[..., -2])
        self.b1 = self.z_max

        self.ldj0 = tf.math.log(self.a0)
        self.ldj1 = tf.math.log(self.a1)

    def _extrapolate_forward(self, y, z):
        # z = a * y + b
        e0 = self.a0 * y + self.b0
        e1 = self.a1 * (y - 1) + self.b1

        z = tf.where(y <= 0, e0, z)
        z = tf.where(y >= 1, e1, z)

        return z

    def _extrapolate_inverse(self, z, y):
        # y = (z - b) / a
        e0 = (z - self.b0) / self.a0
        e1 = (z - self.b1) / self.a1 + 1

        y = tf.where(z <= self.z_min, e0, y)
        y = tf.where(z >= self.z_max, e1, y)

        return y

    def _extrapolate_forward_log_det_jacobian(self, y, ldj):
        # dz = a
        ldj = tf.where(y <= 0.0, self.ldj0, ldj)
        ldj = tf.where(y >= 1.0, self.ldj1, ldj)
        return ldj

    def inverse(self, z: tf.Tensor, **kwds) -> tf.Tensor:
        """
        Returns the inverse Bijector evaluation.

        :param      z:    The input to the inverse evaluation.
        :type       z:    Tensor

        :returns:   The inverse Bijector evaluation.
        :rtype:     Tensor
        """
        y = super().inverse(
            tf.where((z <= self.z_min) | (z >= self.z_max), tf.zeros_like(z), z), **kwds
        )
        return self._extrapolate_inverse(z, y)

    def _forward(self, y: tf.Tensor) -> tf.Tensor:
        """
        Returns the forward Bijector evaluation.

        :param      y:    The input to the forward evaluation.
        :type       y:    Tensor

        :returns:   The forward Bijector evaluation.
        :rtype:     Tensor
        """
        # Note: If the gradient of either branch of the tf.where generates a NaN,
        #       then the gradient of the entire tf.where will be NaN.
        #       -> use an inner tf.where to ensure the function has no asymptote
        z = super()._forward(tf.where((y <= 0) | (y >= 1), tf.ones_like(y), y))

        z = self._extrapolate_forward(y, z)
        return z

    def _forward_log_det_jacobian(self, y):
        ldj = super()._forward_log_det_jacobian(
            tf.where((y <= 0) | (y >= 1), tf.zeros_like(y), y)
        )
        ldj = self._extrapolate_forward_log_det_jacobian(y, ldj)

        return ldj
