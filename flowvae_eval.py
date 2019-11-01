import argparse

import numpy as np
import tensorflow as tf

import datasets
import flow_vae


def main():
    datasets_available = [f[4:] for f in dir(datasets) if f.startswith('get_') and callable(getattr(datasets, f))]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_weights_path')
    argparser.add_argument('--n_repeats', type=int, default=1)

    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--test_iwae_samples', type=int, default=5000)
    argparser.add_argument('--test_iwae_batch_size', type=int, default=None)
    argparser.add_argument('--evaluate_split', choices=['train', 'val', 'test'], default='test')
    argparser.add_argument('--dataset', choices=datasets_available, default='dynamic_mnist')
    argparser.add_argument('--datasets_dir', default='./datasets/')

    flow_vae.utils.add_model_args(argparser)
    args = argparser.parse_args()

    dataset = getattr(datasets, 'get_%s' % args.dataset)(args.datasets_dir)

    print('Aguments:')
    for param_name, param_value in sorted(vars(args).items()):
        print('--{:30}: {}'.format(param_name, param_value))
    print('\n')

    sess = tf.InteractiveSession()
    vae = flow_vae.utils.get_model(args)
    tf.train.Saver().restore(sess, args.model_weights_path)

    # Evaluation
    data = {
        'train': dataset.train,
        'test': dataset.test,
        'val': dataset.validation,
    }[args.evaluate_split]

    print('Evaluating evidence, KLs on {} with M={}'.format(args.evaluate_split, args.test_iwae_samples))
    evidences = flow_vae.utils.calculate_evidence(
        sess, data, vae, args.test_iwae_samples, args.batch_size, args.n_repeats, tqdm_desc='Calculating evidence')
    print('* log p(x) >= {:.5f} (std. {:.5f})'.format(np.mean(evidences), np.std(evidences)))


if __name__ == "__main__":
    main()
