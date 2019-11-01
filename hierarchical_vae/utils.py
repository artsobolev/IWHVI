import numpy as np
import scipy as sp
import scipy.special
from tqdm import tqdm

import utils
from .model import HierarchicalVAE


def calculate_evidence(sess, data, iwhvae, iwae_samples, iwhvi_samples, batch_size, n_repeats,
                       tau_force_prior=False, tqdm_desc=None):
    losses = utils.batched_run(sess, data, -iwhvae.loss, lambda x_batch, y_batch: {
        iwhvae.input_x: x_batch,
        iwhvae.output_y: y_batch,
        iwhvae.k_iwhvi_samples: iwhvi_samples,
        iwhvae.m_iwae_samples: iwae_samples,
        iwhvae.tau_force_prior: tau_force_prior
    }, batch_size, n_repeats, tqdm_desc)

    return np.array(list(losses))


def calculate_reused_sivi_bound(sess, data, iwhvae, iwae_samples, iwhvi_samples, batch_size_x, n_repeats,
                                batch_size_m=None, tqdm_desc=None):
    batch_size_x = min(batch_size_x, data.num_examples)  # Fix
    x_samples = data.num_examples
    total_batches_x = (x_samples - 1) // batch_size_x + 1

    if batch_size_m is None:
        batch_size_m = iwae_samples
    total_batches_m = (iwae_samples - 1) // batch_size_m + 1

    res = np.zeros(n_repeats)
    with tqdm(total=n_repeats * total_batches_x * (total_batches_m + 1), unit='run', desc=tqdm_desc) as tqdm_t:
        for rep in range(n_repeats):
            offset_x = 0
            mean_elbo = 0
            for _ in range(total_batches_x):
                batch = data.next_batch(batch_size_x)
                binarized_batch_x, binarized_batch_y = utils.binarize_batch(batch)
                actual_batch_size_x = binarized_batch_x.shape[0]

                psi_rest = sess.run(iwhvae._reuse_psi_rest, feed_dict={
                    iwhvae.input_x: binarized_batch_x,
                    iwhvae.k_iwhvi_samples: iwhvi_samples
                })
                alphas = np.zeros((iwae_samples, actual_batch_size_x))
                offset_m = 0
                for _ in range(total_batches_m):
                    m_samples = min(iwae_samples - offset_m, batch_size_m)
                    alpha = sess.run(iwhvae._sivi_reused_alpha, {
                        iwhvae.input_x: binarized_batch_x,
                        iwhvae.output_y: binarized_batch_y,
                        iwhvae.m_iwae_samples: m_samples,
                        iwhvae.k_iwhvi_samples: iwhvi_samples,
                        iwhvae._reuse_psi_rest: psi_rest
                    })
                    alphas[offset_m:offset_m + m_samples, :] = alpha
                    offset_m += m_samples
                    tqdm_t.update()

                elbo = np.mean(sp.special.logsumexp(alphas, axis=0) - np.log(iwae_samples))
                offset_x += actual_batch_size_x
                mean_elbo += (elbo - mean_elbo) * actual_batch_size_x / offset_x
                tqdm_t.update()

            res[rep] = mean_elbo
    return res


def batched_calculate_evidence_q_gap_kls(sess, data, iwhvae, iwae_samples, iwhvi_samples, batch_size_x, n_repeats,
                                         tau_force_prior=False, batch_size_m=None, tqdm_desc=None):

    batch_size_x = min(batch_size_x, data.num_examples)  # Fix
    x_samples = data.num_examples
    total_batches_x = (x_samples - 1) // batch_size_x + 1

    if batch_size_m is None:
        batch_size_m = iwae_samples
    total_batches_m = (iwae_samples - 1) // batch_size_m + 1

    aux = [iwhvae._q_gap_raw, *iwhvae.bounds]
    aux_size = len(aux)
    res = np.zeros((1 + aux_size, n_repeats))
    with tqdm(total=n_repeats * total_batches_x, unit='run', desc=tqdm_desc) as tqdm_t:
        for rep in range(n_repeats):
            rep_res = np.zeros((1 + aux_size, x_samples))
            offset_x = 0
            for _ in range(total_batches_x):
                batch = data.next_batch(batch_size_x)
                binarized_batch_x, binarized_batch_y = utils.binarize_batch(batch)
                actual_batch_size_x = binarized_batch_x.shape[0]

                zs = np.zeros((iwae_samples, actual_batch_size_x, iwhvae.z_dim))
                res_aux = np.zeros((aux_size, iwae_samples, actual_batch_size_x))
                offset_m = 0
                for _ in range(total_batches_m):
                    m_samples = min(iwae_samples - offset_m, batch_size_m)
                    aux_vals = sess.run(aux + [iwhvae._z], {
                        iwhvae.input_x: binarized_batch_x,
                        iwhvae.output_y: binarized_batch_y,
                        iwhvae.m_iwae_samples: m_samples,
                        iwhvae.k_iwhvi_samples: iwhvi_samples,
                        iwhvae.tau_force_prior: tau_force_prior
                    })
                    zs[offset_m:offset_m + m_samples] = aux_vals[-1]
                    res_aux[:, offset_m:offset_m + m_samples, :] = aux_vals[:-1]
                    offset_m += m_samples

                elbos = sess.run(iwhvae.elbo, {
                    iwhvae.input_x: binarized_batch_x,
                    iwhvae.output_y: binarized_batch_y,
                    iwhvae.m_iwae_samples: iwae_samples,
                    iwhvae.k_iwhvi_samples: iwhvi_samples,
                    iwhvae.tau_force_prior: tau_force_prior,
                    iwhvae._z: zs,
                    iwhvae.bounds[1]: res_aux[aux.index(iwhvae.bounds[1])]
                })

                rep_res[:, offset_x:offset_x + actual_batch_size_x] = [
                    elbos,
                    *np.mean(res_aux, axis=1)
                ]
                offset_x += actual_batch_size_x
                tqdm_t.update()

            res[:, rep] = np.mean(rep_res, axis=1)
    return res


def batched_calculate_p_gap(sess, data, iwhvae, iwae_samples, iwhvi_samples, batch_size_x,
                            n_repeats, tau_force_prior, batch_size_m=None, tqdm_desc=None):

    batch_size_x = min(batch_size_x, data.num_examples)  # Fix
    x_samples = data.num_examples
    total_batches_x = (x_samples - 1) // batch_size_x + 1

    if batch_size_m is None:
        batch_size_m = iwae_samples
    total_batches_m = (iwae_samples - 1) // batch_size_m + 1

    p_gaps = np.zeros(n_repeats)
    with tqdm(total=n_repeats * total_batches_x, unit='run', desc=tqdm_desc) as tqdm_t:
        for rep in range(n_repeats):
            mean_p_gap = 0
            offset_x = 0
            for _ in range(total_batches_x):
                batch = data.next_batch(batch_size_x)
                binarized_batch_x, binarized_batch_y = utils.binarize_batch(batch)
                actual_batch_size_x = binarized_batch_x.shape[0]

                alphas = np.zeros((iwae_samples, actual_batch_size_x))
                offset_m = 0
                for _ in range(total_batches_m):
                    m_samples = min(iwae_samples - offset_m, batch_size_m)
                    alpha = sess.run(iwhvae._alpha, {
                        iwhvae.input_x: binarized_batch_x,
                        iwhvae.output_y: binarized_batch_y,
                        iwhvae.m_iwae_samples: m_samples,
                        iwhvae.k_iwhvi_samples: iwhvi_samples,
                        iwhvae.tau_force_prior: tau_force_prior
                    })

                    alphas[offset_m:offset_m + m_samples] = alpha
                    offset_m += m_samples

                batch_p_gap = sp.special.logsumexp(alphas, axis=0) - np.log(alphas.shape[0]) - np.mean(alphas, axis=0)
                batch_mean_p_gap = np.mean(batch_p_gap, axis=0)
                offset_x += actual_batch_size_x
                mean_p_gap += (batch_mean_p_gap - mean_p_gap) * actual_batch_size_x / offset_x

                tqdm_t.update()

            p_gaps[rep] = mean_p_gap
    return p_gaps


def calculate_mi_bounds_on_p(sess, vae, n_samples, m_iwae_samples, iwae_batch_size, n_repeats, tqdm_desc):
    res = []
    with tqdm(total=n_repeats * n_samples * (m_iwae_samples + 1), desc=tqdm_desc) as tqdm_t:
        mi_lower_bound_per_sample = []
        mi_upper_bound_per_sample = []

        for _ in range(n_repeats):
            for _ in range(n_samples):
                x, hvm_term, log_p_x_z = sess.run([vae._mi_p_x, vae._mi_p_hvm_term, vae._mi_p_log_p_x_z])
                tqdm_t.update(1)

                elbos = []
                m_offset = 0
                while m_offset < m_iwae_samples:
                    batch_size = min(iwae_batch_size, m_iwae_samples - m_offset)
                    m_offset += batch_size

                    elbos_subset = sess.run(vae._mi_p_elbos, feed_dict={vae._mi_p_x: x, vae.m_iwae_samples: batch_size})
                    elbos.extend(elbos_subset)

                    tqdm_t.update(batch_size)

                assert len(elbos) == m_iwae_samples
                log_p_lower_bound = sp.special.logsumexp(elbos) - np.log(m_iwae_samples)
                log_p_upper_bound = sp.special.logsumexp(elbos + [hvm_term]) - np.log(m_iwae_samples + 1)

                mi_lower_bound_per_sample.append(log_p_x_z - log_p_upper_bound)
                mi_upper_bound_per_sample.append(log_p_x_z - log_p_lower_bound)

            res.append((np.mean(mi_lower_bound_per_sample), np.mean(mi_upper_bound_per_sample)))
            tqdm_t.set_description(tqdm_desc + ', cur. est.: {:.3f} <= MI <= {:.3f}'.format(*np.mean(res, axis=0)))

    return np.array(res).T


def calculate_mi_bounds_on_q(sess, vae, data, batch_size_x, m_iwae_samples, iwae_batch_size, n_repeats, tqdm_desc):
    if iwae_batch_size is None:
        iwae_batch_size = m_iwae_samples

    total_batches_x = (data.num_examples - 1) // batch_size_x + 1
    res = []
    with tqdm(total=n_repeats * data.num_examples * (m_iwae_samples + 1), desc=tqdm_desc) as tqdm_t:
        mi_lower_bound_per_sample = []
        mi_upper_bound_per_sample = []

        for _ in range(n_repeats):
            for _ in range(total_batches_x):
                binarized_x, _ = utils.binarize_batch(data.next_batch(batch_size_x))
                actual_batch_size_x = binarized_x.shape[0]

                z, hvm_term, log_q_z_psi = sess.run([vae._mi_q_z, vae._mi_q_hvm_term, vae._mi_q_log_q_z_psi],
                                                    feed_dict={vae.input_x: binarized_x})
                tqdm_t.update(actual_batch_size_x)

                elbos = np.zeros((m_iwae_samples, actual_batch_size_x))
                m_offset = 0
                while m_offset < m_iwae_samples:
                    batch_size_m = min(iwae_batch_size, m_iwae_samples - m_offset)

                    elbos_subset = sess.run(vae._mi_q_elbos, feed_dict={
                        vae._mi_q_z: z,
                        vae.m_iwae_samples: batch_size_m,
                        vae.input_x: binarized_x
                    })
                    assert elbos_subset.shape == (batch_size_m, actual_batch_size_x), elbos_subset.shape
                    elbos[m_offset:m_offset+batch_size_m, :] = elbos_subset

                    tqdm_t.update(batch_size_m * actual_batch_size_x)
                    m_offset += batch_size_m

                extended_elbos = np.concatenate([hvm_term[None, :], elbos], axis=0)

                assert len(elbos) == m_iwae_samples
                log_p_lower_bound = sp.special.logsumexp(elbos, axis=0) - np.log(m_iwae_samples)
                log_p_upper_bound = sp.special.logsumexp(extended_elbos, axis=0) - np.log(m_iwae_samples + 1)

                mi_lower_bound_per_sample.append(log_q_z_psi - log_p_upper_bound)
                mi_upper_bound_per_sample.append(log_q_z_psi - log_p_lower_bound)

            res.append((np.mean(mi_lower_bound_per_sample), np.mean(mi_upper_bound_per_sample)))
            tqdm_t.set_description(tqdm_desc + ', cur. est.: {:.3f} <= MI <= {:.3f}'.format(*np.mean(res, axis=0)))

    return np.array(res).T


def calculate_kl_tau_q(sess, data, iwhvae, n_samples, batch_size, n_repeats, tau_force_prior=False, tqdm_desc=None):
    kls = utils.batched_run(
        sess, data, iwhvae.kl_tau_q,
        lambda x_batch, y_batch: {
            iwhvae.input_x: x_batch,
            iwhvae.output_y: y_batch,
            iwhvae.m_iwae_samples: n_samples,
            iwhvae.tau_force_prior: tau_force_prior
        },
        batch_size, n_repeats, tqdm_desc)

    return np.array(list(kls))


def calculate_q_gap(sess, data, iwhvae, m_samples, k_iwhvi_samples, batch_size, n_repeats,
                    tau_force_prior=False, tqdm_desc=None):
    gaps = utils.batched_run(sess, data, iwhvae.q_gap, lambda x_batch, y_batch: {
        iwhvae.input_x: x_batch,
        iwhvae.output_y: y_batch,
        iwhvae.m_iwae_samples: m_samples,
        iwhvae.k_iwhvi_samples: k_iwhvi_samples,
        iwhvae.tau_force_prior: tau_force_prior
    }, batch_size, n_repeats, tqdm_desc)

    return np.array(list(gaps))


def calculate_kl_bounds(sess, data, iwhvae, m_samples, k_iwhvi_samples, batch_size, n_repeats,
                        tau_force_prior=False, tqdm_desc=None):
    bounds = utils.batched_run(sess, data, [iwhvae.kl_lower_bound, iwhvae.kl_upper_bound], lambda x_batch, y_batch: {
        iwhvae.input_x: x_batch,
        iwhvae.output_y: y_batch,
        iwhvae.m_iwae_samples: m_samples,
        iwhvae.k_iwhvi_samples: k_iwhvi_samples,
        iwhvae.tau_force_prior: tau_force_prior
    }, batch_size, n_repeats, tqdm_desc)

    avg_bounds = np.array(list(bounds))
    return avg_bounds[:, 0], avg_bounds[:, 1]


def add_model_args(argparser):
    argparser.add_argument('--z_dim', type=int, default=10)
    argparser.add_argument('--noise_dim', type=int, default=10)

    argparser.add_argument('--decoder_arch', type=int, nargs='*', default=[200, 200])
    argparser.add_argument('--encoder_arch', type=int, nargs='*', default=[200, 200])
    argparser.add_argument('--encoder_noise_arch', type=int, nargs='*', default=[])
    argparser.add_argument('--tau_arch', type=int, nargs='*', default=[200, 200])

    argparser.add_argument('--tau_gate_bias', type=float, default=np.nan)
    argparser.add_argument('--encoder_student', action='store_true')
    argparser.add_argument('--tau_student', action='store_true')
    argparser.add_argument('--batch_norm', action='store_true')


def get_model(args):
    return HierarchicalVAE(
        28 * 28, 28 * 28, args.noise_dim, args.z_dim,
        args.decoder_arch, args.encoder_arch, args.encoder_noise_arch, args.tau_arch,
        args.tau_gate_bias, args.encoder_student, args.tau_student, args.batch_norm)
