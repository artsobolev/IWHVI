import numpy as np

import utils
from .model import FlowVAE


def add_model_args(argparser):
    argparser.add_argument('--z_dim', type=int, default=10)

    argparser.add_argument('--decoder_arch', type=int, nargs='*', default=[200, 200])
    argparser.add_argument('--encoder_arch', type=int, nargs='*', default=[200, 200])
    argparser.add_argument('--encoder_flow_arch', type=utils.make_typed_tuple(str, rest=int), nargs='*', default=[])
    argparser.add_argument('--encoder_student', action='store_true')


def get_model(args):
    return FlowVAE(28 * 28, 28 * 28, args.z_dim, args.decoder_arch,
                   args.encoder_arch, args.encoder_flow_arch, args.encoder_student)


def calculate_evidence(sess, data, vae, iwae_samples, batch_size_x, n_repeats, tqdm_desc=None):
    losses = utils.batched_run(sess, data, -vae.loss, lambda x_batch, y_batch: {
        vae.input_x: x_batch,
        vae.output_y: y_batch,
        vae.m_iwae_samples: iwae_samples,
    }, batch_size_x, n_repeats, tqdm_desc)

    return np.array(list(losses))


def calculate_kl_q(sess, data, vae, n_samples, batch_size, n_repeats, tqdm_desc=None):
    kls = utils.batched_run(
        sess, data, vae.q_kl,
        lambda x_batch, y_batch: {
            vae.input_x: x_batch,
            vae.output_y: y_batch,
            vae.m_iwae_samples: n_samples
        },
        batch_size, n_repeats, tqdm_desc)

    return np.array(list(kls))
