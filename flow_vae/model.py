import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp


import utils

eps = 1e-10


class FlowVAE:
    def __init__(self, x_dim, y_dim, z_dim, decoder_arch, encoder_arch, encoder_flow_arch=(), encoder_student=False):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.activation_fn = tf.nn.softplus
        self.decoder_arch = decoder_arch
        self.encoder_arch = encoder_arch
        self.encoder_student = encoder_student
        self.encoder_flow_arch = encoder_flow_arch

        self._build()

    def _build(self):
        self.input_x = tf.placeholder(tf.float32, shape=(None, self.x_dim), name='x')
        self.output_y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y')

        self.kl_coef = tf.placeholder_with_default(1.0, shape=(), name='kl_coef')
        self.m_iwae_samples = tf.placeholder_with_default(1, shape=(), name='m_iwae_samples')

        encoder = self._encoder()
        z = encoder.sample(self.m_iwae_samples)  # M x N x D_z
        log_q_z = encoder.log_prob(z)

        self.decoder = self._decoder(z)  # M x N x D_x
        log_p_y_z = tf.reduce_sum(self.decoder.log_prob(self.output_y), axis=2)  # M x N
        log_p_z = self._prior().log_prob(z)  # M x N

        kl = log_q_z - log_p_z
        elbos = log_p_y_z - self.kl_coef * kl
        self.elbo = tf.reduce_logsumexp(elbos, axis=0) - tf.log(tf.to_float(self.m_iwae_samples))  # N
        self.loss = -tf.reduce_mean(self.elbo, axis=0)

        self.q_kl = tf.reduce_mean(kl, axis=[0, 1])

    def gradients(self, var_list=None, scope=None):
        if var_list is None:
            var_list = slim.get_model_variables(scope=scope)

        grads = tf.gradients(self.loss, var_list)
        return list(zip(grads, var_list))

    def _encoder(self, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('encoder', reuse=reuse):
            net = self.input_x
            for layer_spec in self.encoder_arch:
                net = slim.fully_connected(net, layer_spec, activation_fn=self.activation_fn)

            mu = slim.fully_connected(net, self.z_dim, activation_fn=None)
            std = slim.fully_connected(net, self.z_dim, activation_fn=tf.nn.softplus)
            flow_cond = slim.fully_connected(net, self.z_dim, activation_fn=None)

            if not self.encoder_student:
                encoder = tfp.distributions.MultivariateNormalDiag(mu, eps + std)
            else:
                df = 2 + tf.clip_by_value(slim.fully_connected(net, 1, activation_fn=tf.nn.softplus), 0, 1e3)
                encoder = tfp.distributions.MultivariateStudentTLinearOperator(
                    df, mu, tf.linalg.LinearOperatorDiag(eps + std))

            return self._conditional_flow(encoder, flow_cond, self.encoder_flow_arch, self.z_dim)

    @staticmethod
    def _conditional_flow(distribution, x_cond, arch, dim):
        bijectors = []
        for layer_idx, (layer_type, *layer_spec) in enumerate(arch):
            if layer_type == 'iaf':
                bijector = tfp.bijectors.Invert(
                    tfp.bijectors.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=utils.masked_autoregressive_conditional_template(
                            x_cond, hidden_layers=layer_spec)
                    )
                )
            else:
                bijector = tfp.bijectors.RealNVP(
                    num_masked=dim // 2,
                    shift_and_log_scale_fn=utils.real_nvp_conditional_template(
                        x_cond, hidden_layers=layer_spec, activation=tf.nn.softplus)
                )

            if layer_idx + 1 < len(arch):
                permutation = tf.get_variable('permutation_%d' % layer_idx, trainable=False, dtype=tf.int32,
                                              initializer=np.random.permutation(dim).astype("int32"))
                bijector = tfp.bijectors.Chain(bijectors=[bijector, tfp.bijectors.Permute(permutation=permutation)])

            bijectors.append(bijector)

        if not bijectors:
            return distribution

        return tfp.distributions.TransformedDistribution(
            distribution=distribution, bijector=tfp.bijectors.Chain(bijectors))

    def _decoder(self, z, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            net = slim.stack(z, slim.fully_connected, self.decoder_arch, activation_fn=self.activation_fn)
            logits_y = slim.fully_connected(net, self.y_dim, activation_fn=None)
            return tfp.distributions.Bernoulli(logits=logits_y)

    def _prior(self):
        return tfp.distributions.MultivariateNormalDiag(tf.zeros(self.z_dim), tf.ones(self.z_dim))
