import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp


from . import dreg_hacks
import utils

eps = 1e-10


class HierarchicalVAE:
    def __init__(self, x_dim, y_dim, psi_dim, z_dim, decoder_arch, encoder_arch, encoder_noise_arch, tau_arch,
                 tau_gate_bias=np.nan, encoder_student=False, tau_student=False, batch_norm=False):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.psi_dim = psi_dim
        self.z_dim = z_dim

        self.tau_gate_bias_ = tau_gate_bias
        self.tau_arch = tau_arch
        self.encoder_arch = encoder_arch
        self.encoder_noise_arch = encoder_noise_arch
        self.decoder_arch = decoder_arch
        self.batch_norm = batch_norm
        self.activation_fn = tf.nn.softplus

        self.encoder_student = encoder_student
        self.tau_student = tau_student

        self._build()

    def _build(self):
        self.input_x = tf.placeholder(tf.float32, shape=(None, self.x_dim), name='x')
        self.output_y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y')

        self.kl_coef = tf.placeholder_with_default(1.0, shape=(), name='kl_coef')
        self.kl_free_bits = tf.placeholder_with_default(-np.inf, shape=(), name='kl_free_bits')
        self.k_iwhvi_samples = tf.placeholder_with_default(1, shape=(), name='k_iwhvi_samples')
        self.m_iwae_samples = tf.placeholder_with_default(1, shape=(), name='m_iwae_samples')
        self.iwhvi_kl_coef = tf.placeholder_with_default(1.0, shape=(), name='iwhvi_kl_coef')
        self.iwhvi_kl_free_bits = tf.placeholder_with_default(-np.inf, shape=(), name='iwhvi_free_bits')
        self.tau_force_prior = tf.placeholder_with_default(False, shape=(), name='tau_force_prior')

        z, psi0, (kl_lower_bound, kl_upper_bound), \
        (tau_psi, (beta_lower, psi_lower, _), (beta_upper, psi_upper, q_z_psi)) = \
            self._build_kl_bounds(self.m_iwae_samples, self.k_iwhvi_samples)

        self.decoder = self._decoder(z)  # M x N x D_x
        log_p_y_z = tf.reduce_sum(self.decoder.log_prob(self.output_y), axis=2)  # M x N

        alpha = log_p_y_z - self.kl_coef * tf.maximum(self.kl_free_bits, kl_upper_bound)
        self.elbo = tf.reduce_logsumexp(alpha, axis=0) - tf.log(tf.to_float(self.m_iwae_samples))  # N
        self.loss = -tf.reduce_mean(self.elbo, axis=0)

        # SIVI reused bound
        reused_kl_bound = self._build_reused_sivi_bound(z, psi0)
        self._sivi_reused_alpha = reused_alpha = log_p_y_z - reused_kl_bound * self.kl_coef
        reused_elbo = tf.reduce_logsumexp(reused_alpha, axis=0) - tf.log(tf.to_float(self.m_iwae_samples))  # N
        self.reused_sivi_avg_elbo = tf.reduce_mean(reused_elbo, axis=0)

        # Save useful variables
        # `kl_upper_bound` has to be the same node as in the ELBO def.
        self.bounds = [kl_lower_bound, kl_upper_bound]
        self._q_z_psi = q_z_psi
        self._tau_psi = tau_psi
        self._alpha = alpha
        self._beta = beta_upper
        self._psi = psi_upper
        self._z = z
        self._psi_true = psi0
        self._log_p_y_z = log_p_y_z

        # Build diagnostics
        self.kl_upper_bound = tf.reduce_mean(kl_upper_bound, axis=[0, 1])
        self.kl_lower_bound = tf.reduce_mean(kl_lower_bound, axis=[0, 1])

        # TODO: should we be looking on the upper gap?
        q_gap = tf.reduce_logsumexp(beta_lower, axis=0) \
                - tf.log(tf.to_float(self.k_iwhvi_samples)) \
                - tf.reduce_mean(beta_lower, axis=0)
        self._q_gap_raw = q_gap
        self.q_gap = tf.reduce_mean(q_gap, axis=[0, 1])

        self._p_gap_raw = self.elbo - tf.reduce_mean(alpha, axis=0)
        self.p_gap = tf.reduce_mean(self._p_gap_raw, axis=0)

        q_psi = self._q_prior()
        try:
            total_kl = tfp.distributions.kl_divergence(tau_psi, q_psi)
        except NotImplementedError:
            n_kl_samples = 100
            tau_samples = tau_psi.sample(n_kl_samples)
            total_kl = tf.reduce_mean(tau_psi.log_prob(tau_samples) - q_psi.log_prob(tau_samples), axis=0)
        self.kl_tau_q = tf.reduce_mean(total_kl, axis=[0, 1])

        self._build_mi_bounds_on_p()
        self._build_mi_bounds_on_q()

    def _build_sivi_gradients(self):
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'q_decoder')
        return list(zip(tf.gradients(-self.reused_sivi_avg_elbo, q_vars), q_vars))

    def build_uivi_surrogate_loss(self, mcmc_samples, mcmc_burnin, mcmc_thinning, mcmc_parallel, mcmc_target_accept_prob,
                                  mcmc_adaptation_rate, hmc_stepsize, hmc_leapfrog_steps):
        z = self._z[0]
        psi0 = self._psi_true[0]  # N x Dpsi

        def _q_psi_z_hmc_target(psi):
            return self._q_decoder(psi).log_prob(z) + self._q_prior().log_prob(psi)

        log_p_joint = self._log_p_y_z[0] + self.kl_coef * self._prior().log_prob(z)  # N
        uivi_hmc_psi0, _ = tfp.mcmc.sample_chain(
            1,
            tf.tile(psi0[None], [mcmc_samples, 1, 1]),   # L x N x Dpsi
            kernel=tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=_q_psi_z_hmc_target,
                    step_size=hmc_stepsize,
                    num_leapfrog_steps=hmc_leapfrog_steps
                ),
                num_adaptation_steps=mcmc_burnin,
                target_accept_prob=mcmc_target_accept_prob,
                adaptation_rate=mcmc_adaptation_rate
            ),
            num_burnin_steps=mcmc_burnin,
            num_steps_between_results=mcmc_thinning,
            parallel_iterations=mcmc_parallel)  # 1 x L x N
        uivi_hmc_psi0 = tf.stop_gradient(tf.squeeze(uivi_hmc_psi0, axis=0))
        uivi_log_q_surrogate = self._q_decoder(uivi_hmc_psi0, stop_gradient=True).log_prob(z)  # L x N
        uivi_loss = -tf.reduce_mean(log_p_joint - self.kl_coef * tf.reduce_mean(uivi_log_q_surrogate, axis=0), axis=0)

        return uivi_loss

    def _build_mi_bounds_on_q(self):
        N = tf.shape(self.input_x)[0]
        q_prior = self._q_prior()
        psi_0 = q_prior.sample(N)  # N x Dpsi
        q_decoder_0 = self._q_decoder(psi_0)
        z = q_decoder_0.sample()  # N x Dz

        log_q_z_psi = q_decoder_0.log_prob(z)

        tau = self._tau_proposal(z)
        psi_1m = tau.sample(self.m_iwae_samples)
        q_decoder_1m = self._q_decoder(psi_1m)

        self._mi_q_z = z
        self._mi_q_log_q_z_psi = log_q_z_psi
        self._mi_q_hvm_term = log_q_z_psi + q_prior.log_prob(psi_0) - tau.log_prob(psi_0)
        self._mi_q_elbos = q_decoder_1m.log_prob(z) + q_prior.log_prob(psi_1m) - tau.log_prob(psi_1m)

    def _build_mi_bounds_on_p(self):
        p_prior = self._prior()
        z_0 = p_prior.sample(1)  # 1 x Dz
        p_decoder_0 = self._decoder(z_0)
        x = tf.to_float(p_decoder_0.sample())  # 1 x Dx

        q_prior = self._q_prior()
        psi_1m = q_prior.sample((self.m_iwae_samples, 1))  # M x 1 x Dpsi
        q_decoder_1m = self._q_decoder(psi_1m, x)
        z_1m = q_decoder_1m.sample()
        tau_1m = self._tau_proposal(z_1m, x)
        p_decoder_1m = self._decoder(z_1m)

        nom_1m = p_prior.log_prob(z_1m) + tf.reduce_sum(p_decoder_1m.log_prob(x), axis=2)
        denom_1m = q_decoder_1m.log_prob(z_1m) + q_prior.log_prob(psi_1m) - tau_1m.log_prob(psi_1m)

        tau_0 = self._tau_proposal(z_0, x)
        psi_0 = tau_0.sample()
        q_decoder_0 = self._q_decoder(psi_0, x)

        log_p_x_z = tf.reduce_sum(p_decoder_0.log_prob(x), axis=1)  # 1
        nom_0 = p_prior.log_prob(z_0) + log_p_x_z
        denom_0 = q_decoder_0.log_prob(z_0) + q_prior.log_prob(psi_0) - tau_0.log_prob(psi_0)

        self._mi_p_x = x
        self._mi_p_log_p_x_z = tf.squeeze(log_p_x_z, axis=0)
        self._mi_p_hvm_term = tf.squeeze(nom_0 - denom_0, axis=0)
        self._mi_p_elbos = tf.squeeze(nom_1m - denom_1m, axis=1)  # M

    def _build_reused_sivi_bound(self, z, psi_true):
        N = tf.shape(self.input_x)[0]
        M = self.m_iwae_samples
        K = self.k_iwhvi_samples

        # z, psi_true: M x N x D
        # psi_rest: K x N x D
        psi_rest = self._q_prior().sample((K, N))  # K x N x D
        self._reuse_psi_rest = psi_rest

        log_q_z_rest = self._q_decoder(psi_rest[:, None]).log_prob(z[None])  # K x M x N
        log_q_z_true = self._q_decoder(psi_true).log_prob(z)  # M x N

        total_log_q_z = tf.concat([log_q_z_true[None], log_q_z_rest], axis=0)  # (K+1) x M x N
        q_approx = tf.reduce_logsumexp(total_log_q_z, axis=0) - tf.log(tf.to_float(K + 1))  # M x N
        log_p_z = self._prior().log_prob(z)

        return q_approx - log_p_z

    def _build_kl_bounds(self, m_samples, k_tau_samples):
        batch_size = tf.shape(self.input_x)[0]

        q_psi = self._q_prior()
        psi_true = q_psi.sample((m_samples, batch_size))  # M x N x D_psi
        self.encoder = self._q_decoder(psi_true)
        z = self.encoder.sample()  # M x N x D_z

        tau_psi = self._tau_proposal(z)
        psi_lower = tau_psi.sample(k_tau_samples)
        log_tau_psi_lower = tau_psi.log_prob(psi_lower, name='tau_dreg_log_prob')  # K x M x N

        log_p_z = self._prior().log_prob(z)  # M x N

        bounds = []
        bounds_data = []
        for psi_extra in [None, psi_true]:
            psi = psi_lower
            if psi_extra is not None:
                psi = tf.concat([psi_extra[None], psi], axis=0)  # K+1 x M x N x D_psi

            q_z_psi_all = self._q_decoder(psi)
            log_q_dec = q_z_psi_all.log_prob(z)  # K x M x N
            log_q_prior = q_psi.log_prob(psi)  # K x M x N
            log_tau_psi = log_tau_psi_lower
            if psi_extra is not None:
                log_tau_psi = tf.concat([tau_psi.log_prob(psi_extra[None]), log_tau_psi], axis=0)

            iwhvi_kl = log_tau_psi - log_q_prior
            beta = log_q_dec - self.iwhvi_kl_coef * tf.maximum(self.iwhvi_kl_free_bits, iwhvi_kl)

            k_bound_samples = tf.shape(psi)[0]
            log_q_z_bound = tf.reduce_logsumexp(beta, axis=0) - tf.log(tf.to_float(k_bound_samples))

            bounds.append(log_q_z_bound - log_p_z)
            bounds_data.append((beta, psi, q_z_psi_all))

        return z, psi_true, bounds, (tau_psi, *bounds_data)

    def build_iwhvi_gradients(self, tau_use_dreg=False, var_list=None, scope=None):
        if var_list is None:
            var_list = slim.get_model_variables(scope=scope)

        filtered_var_list = var_list
        if tau_use_dreg:
            # We only do DReG w.r.t. tau to ensure fair comparison to prior work
            filtered_var_list = [v for v in filtered_var_list if not v.name.startswith('tau/')]

        grads_and_vars = []
        if filtered_var_list:
            grads = tf.gradients(self.loss, filtered_var_list)
            grads_and_vars.extend(zip(grads, filtered_var_list))

        if tau_use_dreg:
            grads_and_vars.extend((grad, var)
                                  for grad, var in self._dreg_tau_gradients()
                                  if var in var_list and grad is not None)

        return grads_and_vars

    def _dreg_tau_gradients(self):
        # WARNING: some terrible code ahead! I warned you
        # alpha: M x N
        # beta: K+1 x M x N
        with tf.name_scope('tau_dreg_lvl1'):
            alpha_softmax = tf.nn.softmax(self._alpha, axis=0)
            beta_softmax = tf.nn.softmax(self._beta, axis=0)
            beta = self._beta
            psi = self._psi
            z = self._z

            tau_vars = slim.get_model_variables(scope='tau/')

            # Heavy dark magic
            stop_grad_tau_beta = tf.contrib.graph_editor.graph_replace(beta, {
                self._tau_psi.hacked_loc: tf.stop_gradient(self._tau_psi.hacked_loc, name='tau_dreg_hacked_loc'),
                self._tau_psi.hacked_scale_diag: tf.stop_gradient(self._tau_psi.hacked_scale_diag,
                                                                  name='tau_dreg_hacked_scale_diag'),
            })

            beta_softmax1 = beta_softmax[1:]
            term_1_coefs_inner = 1 - beta_softmax1 * (self.kl_coef * (1 - alpha_softmax) + 1)[None]
            term_1_coefs = self.kl_coef * alpha_softmax * beta_softmax1 * (self.iwhvi_kl_coef * term_1_coefs_inner - 1)
            term_1 = tf.stop_gradient(term_1_coefs) * stop_grad_tau_beta[1:]
            term_2 = self.kl_coef * tf.stop_gradient(alpha_softmax * beta_softmax[0]) * -beta[0]

            loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(term_1, axis=0) + term_2, axis=0), axis=0),
                               name='tau_dreg_surrogate_loss')
            dreg_debug_a = -tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(term_1, axis=0), axis=0), axis=0)
            dreg_debug_b = -tf.reduce_mean(tf.reduce_sum(term_2, axis=0), axis=0)

            self._dreg_debug_a_grads = list(zip(tf.gradients(dreg_debug_a, tau_vars), tau_vars))
            self._dreg_debug_b_grads = list(zip(tf.gradients(dreg_debug_b, tau_vars), tau_vars))

            grads = tf.gradients(loss, tau_vars)
            return list(zip(grads, tau_vars))

    def _q_decoder(self, psi, x=None, reuse=tf.AUTO_REUSE, stop_gradient=False):
        if x is None:
            x = self.input_x

        with tf.variable_scope('q_decoder', reuse=reuse):
            with tf.variable_scope('q_psi_net', reuse=reuse):
                psi = slim.stack(psi, slim.fully_connected, self.encoder_noise_arch, activation_fn=self.activation_fn,
                                 normalizer_fn=slim.batch_norm if self.batch_norm else None)

            net = x
            for layer_spec in self.encoder_arch:
                # inp = tf.concat(utils.tile_to_common_shape(net, psi), axis=-1)
                h_net = slim.fully_connected(net, layer_spec, activation_fn=None,
                                             normalizer_fn=slim.batch_norm if self.batch_norm else None)
                h_psi = slim.fully_connected(psi, layer_spec, activation_fn=None,
                                             normalizer_fn=slim.batch_norm if self.batch_norm else None)
                net = self.activation_fn(sum(utils.tile_to_common_shape(h_net, h_psi)))

            net = tf.concat(utils.tile_to_common_shape(net, psi), axis=-1)
            mu = slim.fully_connected(net, self.z_dim, activation_fn=None)
            std = slim.fully_connected(net, self.z_dim, activation_fn=tf.nn.softplus)

            if stop_gradient:
                mu = tf.stop_gradient(mu)
                std = tf.stop_gradient(std)

            if not self.encoder_student:
                q_decoder = dreg_hacks.HackedMultivariateNormalDiag(mu, eps + std, name='q_hackednormal')
            else:
                df = 2 + tf.clip_by_value(slim.fully_connected(net, self.z_dim, activation_fn=tf.nn.softplus), 0, 1e3)
                if stop_gradient:
                    df = tf.stop_gradient(stop_gradient)
                q_decoder = dreg_hacks.HackedStudentT(df, mu, eps + std, name='q_hackedstudent')

            return q_decoder

    def _tau_proposal(self, z, x=None, reuse=tf.AUTO_REUSE):
        if x is None:
            x = self.input_x

        with tf.variable_scope('tau', reuse=reuse):
            # z: M x N x D
            first_layer_spec, *rest_tau_arch = self.tau_arch
            h_x = slim.fully_connected(x, first_layer_spec, activation_fn=None,
                                       normalizer_fn=slim.batch_norm if self.batch_norm else None)
            h_z = slim.fully_connected(z, first_layer_spec, activation_fn=None,
                                       normalizer_fn=slim.batch_norm if self.batch_norm else None)
            net = self.activation_fn(sum(utils.tile_to_common_shape(h_x, h_z)))
            if rest_tau_arch:
                net = slim.stack(net, slim.fully_connected, rest_tau_arch,
                                 activation_fn=self.activation_fn,
                                 normalizer_fn=slim.batch_norm if self.batch_norm else None)

            # weights_initializer = tf.initializers.random_normal(0., 1e-3)
            mu = slim.fully_connected(net, self.psi_dim, activation_fn=None)
            std = slim.fully_connected(net, self.psi_dim, activation_fn=tf.nn.softplus)

            if not np.isnan(self.tau_gate_bias_):
                adaptive_gate = slim.fully_connected(net, self.psi_dim, activation_fn=tf.nn.sigmoid,
                                                     biases_initializer=tf.initializers.constant(self.tau_gate_bias_))

                mu = mu * adaptive_gate
                std = 1.0 + adaptive_gate * (std - 1.0)

            hard_gate = tf.to_float(tf.logical_not(self.tau_force_prior))
            mu = mu * hard_gate
            std = 1.0 + hard_gate * (std - 1.0)

            # Some minor dark magic for DReG
            if self.tau_student:
                raw_df = slim.fully_connected(net, self.psi_dim, activation_fn=tf.nn.softplus)
                df = 2 + tf.clip_by_value(raw_df, 0, 1e3)
                tau = dreg_hacks.HackedStudentT(df, mu, eps + std, name='tau_hackedstudent')
            else:
                tau = dreg_hacks.HackedMultivariateNormalDiag(mu, eps + std, name='tau_hackednormal')

            return tau

    def _q_prior(self):
        loc = tf.zeros(self.psi_dim)
        scale = tf.ones(self.psi_dim)
        return tfp.distributions.MultivariateNormalDiag(loc, scale)

    def _decoder(self, z, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("decoder", reuse=reuse):
            net = slim.stack(z, slim.fully_connected, self.decoder_arch, activation_fn=self.activation_fn)
            logits_y = slim.fully_connected(net, self.y_dim, activation_fn=None)
            return tfp.distributions.Bernoulli(logits=logits_y)

    def _prior(self):
        return tfp.distributions.MultivariateNormalDiag(tf.zeros(self.z_dim), tf.ones(self.z_dim))
