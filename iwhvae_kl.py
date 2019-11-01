import argparse
import pickle

from tqdm import tqdm
import numpy as np
import tensorflow as tf

import datasets
import hierarchical_vae
import utils


def main():
    datasets_available = [f[4:] for f in dir(datasets) if f.startswith('get_') and callable(getattr(datasets, f))]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_weights_path')

    argparser.add_argument('--max_test_batch_size', type=int, default=1)
    argparser.add_argument('--test_iwae_samples', type=int, default=5000)
    argparser.add_argument('--test_iwae_batch_size', type=int, default=None)
    argparser.add_argument('--test_iwhvi_samples', type=int, nargs='+', default=[0, 1, 10, 25, 50, 100, 200])
    argparser.add_argument('--diagnostic_kl_batch_size', type=int, default=10)
    argparser.add_argument('--evaluate_split', choices=['train', 'val', 'test'], default='test')
    argparser.add_argument('--dataset', choices=datasets_available, default='dynamic_mnist')
    argparser.add_argument('--datasets_dir', default='./datasets/')

    hierarchical_vae.utils.add_model_args(argparser)
    args = argparser.parse_args()

    dataset = getattr(datasets, 'get_%s' % args.dataset)(args.datasets_dir)

    sess = tf.InteractiveSession()

    print('Aguments:')
    for param_name, param_value in sorted(vars(args).items()):
        print('--{:30}: {}'.format(param_name, param_value))
    print('\n')

    vae = hierarchical_vae.utils.get_model(args)
    sess.run(tf.global_variables_initializer())
    restorer = tf.train.Saver()
    restorer.restore(sess, args.model_weights_path)

    # Evaluation
    data = {
        'train': dataset.train,
        'test': dataset.test,
        'val': dataset.validation,
    }[args.evaluate_split]

    p_gaps = {}
    for iwhvi_samples in args.test_iwhvi_samples:
        x_batch_size = args.max_test_batch_size // (iwhvi_samples + 1) + 1
        print('Evaluating evidence, KLs and q gap with {} on {} with M={}, K={}, batch_size={}'.format(
            'IWHVI', args.evaluate_split, args.test_iwae_samples, iwhvi_samples, x_batch_size))
        p_gaps = hierarchical_vae.utils.batched_calculate_p_gap(
            sess, data, vae, args.test_iwae_samples, iwhvi_samples, x_batch_size,
            args.n_repeats, False, args.test_iwae_batch_size, tqdm_desc='Calculating log p(x) gap')

        p_gaps[iwhvi_samples] = p_gaps

    print('*' * 80)
    print('* {} log q(z) gap (using {} bound)'.format(args.evaluate_split, 'IWHVI'))
    for k in args.test_iwhvi_samples:
        print('* k = {:4} * log q(z) gap is {:.5f} (std.: {:.5f})'.format(k, np.mean(p_gaps[k]), np.std(p_gaps[k])))
    print('*' * 80)
    print()

    print('Evaluating KL(tau||q) on {} with M={}'.format(args.evaluate_split, args.test_iwae_samples))
    kls_tau_q = hierarchical_vae.utils.calculate_kl_tau_q(
        sess, data, vae, args.test_iwae_samples, args.diagnostic_kl_batch_size, args.n_repeats,
        tqdm_desc='Calculating KL(tau(psi)||q(psi))')

    print('* The final KL(tau(psi)||q(psi)) on {}: {:.5f} (std.: {:.5f})'.format(
        args.evaluate_split, np.mean(kls_tau_q), np.std(kls_tau_q)))
    print('*' * 80)


if __name__ == "__main__":
    main()
