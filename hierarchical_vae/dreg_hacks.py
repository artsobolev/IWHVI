import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribution_util


class HackedNormal(tfp.distributions.Normal):
    """
    This is usual Normal distribution except you can stop gradient
    hacked_mu and hacked_sigma to zero out partial derivative of
    the log_prob w.r.t. mu and sigma
    """
    def __init__(self, *args, **kwargs):
        tfp.distributions.Normal.__init__(self, *args, **kwargs)

        # These are used for stop_gradient
        self.hacked_loc = tf.identity(self.loc, name='hacked_loc')
        self.hacked_scale = tf.identity(self.scale, name='hacked_scale')

    def _log_prob(self, x):
        z = (x - self.hacked_loc) / self.hacked_scale
        return -0.5 * tf.square(z) - 0.5 * tf.log(2. * np.pi) - tf.log(self.hacked_scale)


class HackedMultivariateNormalDiag(tfp.distributions.MultivariateNormalDiag):
    def __init__(self, loc, scale_diag, *args, **kwargs):
        super(HackedMultivariateNormalDiag, self).__init__(loc, scale_diag, *args, **kwargs)
        self.hacked_loc = tf.identity(loc, name='hacked_loc')
        self.hacked_scale_diag = tf.identity(scale_diag, name='hacked_scale_diag')
        hacked_scale = distribution_util.make_diag_scale(
            loc=self.hacked_loc,
            scale_diag=self.hacked_scale_diag,
            scale_identity_multiplier=None,
            validate_args=False,
            assert_positive=False
        )

        self._hacked_bijector = tfp.bijectors.AffineLinearOperator(shift=self.hacked_loc, scale=hacked_scale)

    def _log_prob(self, y):
        x = self._hacked_bijector.inverse(y)
        event_ndims = self._maybe_get_static_event_ndims()

        ildj = self._hacked_bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims)
        if self._hacked_bijector._is_injective:
            return self._finish_log_prob_for_one_fiber(y, x, ildj, event_ndims)

        lp_on_fibers = [
            self._finish_log_prob_for_one_fiber(y, x_i, ildj_i, event_ndims)
            for x_i, ildj_i in zip(x, ildj)]
        return tf.reduce_logsumexp(tf.stack(lp_on_fibers), axis=0)


class HackedStudentT(tfp.distributions.StudentT):
    """
    Same as HackedNormal but for StudentT
    """
    def __init__(self, *args, **kwargs):
        tfp.distributions.StudentT.__init__(self, *args, **kwargs)

        # These are used for stop_gradient
        self.hacked_df = tf.identity(self.df, name='hacked_loc')
        self.hacked_loc = tf.identity(self.loc, name='hacked_loc')
        self.hacked_scale = tf.identity(self.scale, name='hacked_scale')

    def _log_prob(self, x):
        y = (x - self.hacked_loc) / self.hacked_scale  # Abs(scale) superfluous.
        log_normalization = (
                tf.log(tf.abs(self.hacked_scale)) + 0.5 * tf.log(self.hacked_df) + 0.5 * np.log(np.pi) +
                tf.math.lgamma(0.5 * self.hacked_df) - tf.math.lgamma(0.5 * (self.hacked_df + 1.))
        )
        return -0.5 * (self.hacked_df + 1.) * tf.log1p(y**2. / self.hacked_df) - log_normalization
