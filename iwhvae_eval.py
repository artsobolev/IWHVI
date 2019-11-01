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
    argparser.add_argument('--n_repeats', type=int, default=1)
    argparser.add_argument('--annealing_factor', type=float, default=1.0)
    argparser.add_argument('--annealing_speed', type=int, default=100)
    argparser.add_argument('--tau_extra_steps', type=int, default=0)

    argparser.add_argument('--finetune_learning_rate', type=float, default=1e-3)
    argparser.add_argument('--finetune_tau_use_dreg', action='store_true')
    argparser.add_argument('--finetune_epochs', type=int, default=0)
    argparser.add_argument('--finetune_iwae_samples', type=int, default=1)
    argparser.add_argument('--finetune_iwhvi_samples', type=int, default=1)
    argparser.add_argument('--finetune_batch_size', type=int, default=256)
    argparser.add_argument('--finetune_q', action='store_true')
    argparser.add_argument('--finetune_reset_tau', action='store_true')
    argparser.add_argument('--finetune_reset_q', action='store_true')
    argparser.add_argument('--save_path', default=None)

    argparser.add_argument('--max_test_batch_size', type=int, default=1)
    argparser.add_argument('--test_iwae_samples', type=int, default=5000)
    argparser.add_argument('--test_iwae_batch_size', type=int, default=None)
    argparser.add_argument('--test_iwhvi_samples', type=int, nargs='+', default=[0, 1, 10, 25, 50, 100, 200])
    argparser.add_argument('--diagnostic_kl_batch_size', type=int, default=10)
    argparser.add_argument('--evaluate_every', type=int, default=1)
    argparser.add_argument('--evaluate_split', choices=['train', 'val', 'test'], default='test')
    argparser.add_argument('--dataset', choices=datasets_available, default='dynamic_mnist')
    argparser.add_argument('--datasets_dir', default='./datasets/')
    argparser.add_argument('--evaluate_equisample', action='store_true')
    argparser.add_argument('--evaluate_equicomp', action='store_true')
    argparser.add_argument('--dump_eval_data_path', default=None)

    hierarchical_vae.utils.add_model_args(argparser)
    args = argparser.parse_args()

    dataset = getattr(datasets, 'get_%s' % args.dataset)(args.datasets_dir)
    train_data = dataset.train
    val_data = dataset.validation

    sess = tf.InteractiveSession()

    print('Aguments:')
    for param_name, param_value in sorted(vars(args).items()):
        print('--{:30}: {}'.format(param_name, param_value))
    print('\n')

    vae = hierarchical_vae.utils.get_model(args)

    lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    tau_gradients = vae.build_iwhvi_gradients(tau_use_dreg=args.finetune_tau_use_dreg, scope='tau/')
    gradients = tau_gradients.copy()
    if args.finetune_q:
        gradients += vae.build_iwhvi_gradients(scope='q_decoder/')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    finetune_op = optimizer.apply_gradients(gradients)
    tau_finetune_op = optimizer.apply_gradients(tau_gradients)

    sess.run(tf.global_variables_initializer())
    restoreable_objects = [
        o for o in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        if not (args.finetune_reset_tau and o.name.startswith('tau/'))
           and not (args.finetune_reset_q and o.name.startswith('q_decoder/'))
    ]
    restorer = tf.train.Saver(var_list=restoreable_objects)
    restorer.restore(sess, args.model_weights_path)

    saver = tf.train.Saver(max_to_keep=50)

    # Fine-tuning
    try:
        for epoch in tqdm(range(args.finetune_epochs), unit='epoch', desc='Fine-tuning tau'):
            np_lr = args.finetune_learning_rate * args.annealing_factor ** (epoch / args.annealing_speed)

            for train_batch in utils.batched_dataset(train_data, args.finetune_batch_size):
                binarized_train_batch_x, binarized_train_batch_y = utils.binarize_batch(train_batch)
                sess.run(finetune_op, {
                    vae.input_x: binarized_train_batch_x,
                    vae.output_y: binarized_train_batch_y,
                    vae.k_iwhvi_samples: args.finetune_iwhvi_samples,
                    vae.m_iwae_samples: args.finetune_iwae_samples,
                    lr: np_lr
                })

                for _ in range(args.tau_extra_steps):
                    sess.run(tau_finetune_op, {
                        vae.input_x: binarized_train_batch_x,
                        vae.output_y: binarized_train_batch_y,
                        vae.k_iwhvi_samples: args.finetune_iwhvi_samples,
                        vae.m_iwae_samples: args.finetune_iwae_samples,
                        lr: np_lr
                    })

            if epoch % args.evaluate_every == 0:
                avg_evidence_train, = hierarchical_vae.utils.calculate_evidence(
                    sess, train_data, vae, args.finetune_iwae_samples,
                    args.finetune_iwhvi_samples, args.finetune_batch_size, 1)
                avg_evidence_test, = hierarchical_vae.utils.calculate_evidence(
                    sess, val_data, vae, args.finetune_iwae_samples,
                    args.finetune_iwhvi_samples, args.finetune_batch_size, 1)

                kl_tau_q, = hierarchical_vae.utils.calculate_kl_tau_q(
                    sess, val_data, vae, args.finetune_iwae_samples, args.finetune_batch_size, 1)

                utils.print_over(
                    "Fine tune epoch: {:4}, "
                    "train_evidence: {:.5f}, "
                    "val_evidence: {:.5f}, "
                    "KL(tau(psi|z)||q(psi)) = {:.5f}".format(epoch, avg_evidence_train, avg_evidence_test, kl_tau_q)
                )

            if args.save_path is not None and epoch % ((args.finetune_epochs - 1) // 100 + 1) == 0 and epoch:
                utils.save_weights(sess, args.save_path, suffix='fine-tuning-ep%d' % epoch, saver=saver)

    except KeyboardInterrupt:
        print('Manual stop')

    if args.save_path is not None:
        utils.save_weights(sess, args.save_path, suffix='fine-tuned', saver=saver)

    # Evaluation
    data = {
        'train': dataset.train,
        'test': dataset.test,
        'val': dataset.validation,
    }[args.evaluate_split]

    eval_data = {
        method_name: {
            k: dict(
                evidences=np.zeros(args.n_repeats),
                q_gaps=np.zeros(args.n_repeats),
                kl_upper_bounds=np.zeros(args.n_repeats),
                kl_lower_bounds=np.zeros(args.n_repeats),
            )
            for k in args.test_iwhvi_samples
        }
        for method_name in ['IWHVI', 'SIVI-like', 'SIVI Equicomp', 'SIVI Equisample']
    }

    # With sample reuse
    if args.evaluate_equisample:
        method_name = 'SIVI Equisample'
        for test_iwhvi_samples in args.test_iwhvi_samples:
            x_batch_size = args.max_test_batch_size // (test_iwhvi_samples + 1) + 1
            print('Evaluating evidence, KLs and q gap with {} on {} with M={}, K={}, batch_size={}'.format(
                method_name, args.evaluate_split, args.test_iwae_samples, test_iwhvi_samples, x_batch_size))
            evidences = hierarchical_vae.utils.calculate_reused_sivi_bound(
                sess, data, vae, args.test_iwae_samples, args.test_iwae_samples * test_iwhvi_samples,
                x_batch_size, args.n_repeats, batch_size_m=args.test_iwae_batch_size, tqdm_desc='Calculating')

            eval_data[method_name][test_iwhvi_samples]['evidences'][:] = evidences

        print('* {} evidence (using {} bound)'.format(args.evaluate_split, method_name))
        for k in args.test_iwhvi_samples:
            evidences = eval_data[method_name][k]['evidences']
            print('* k = {:4} * {:.5f} (std. {:.5f})'.format(k, np.mean(evidences), np.std(evidences)))

    # With sample reuse, but same compute
    if args.evaluate_equicomp:
        method_name = 'SIVI Equicomp'
        for test_iwhvi_samples in args.test_iwhvi_samples:
            x_batch_size = args.max_test_batch_size // (test_iwhvi_samples + 1) + 1
            print('Evaluating evidence, KLs and q gap with {} on {} with M={}, K={}, batch_size={}'.format(
                method_name, args.evaluate_split, args.test_iwae_samples, test_iwhvi_samples, x_batch_size))
            evidences = hierarchical_vae.utils.calculate_reused_sivi_bound(
                sess, data, vae, args.test_iwae_samples, test_iwhvi_samples,
                x_batch_size, args.n_repeats, tqdm_desc='Calculating')

            eval_data[method_name][test_iwhvi_samples]['evidences'][:] = evidences

        print('* {} evidence (using {} bound)'.format(args.evaluate_split, method_name))
        for k in args.test_iwhvi_samples:
            evidences = eval_data[method_name][k]['evidences']
            print('* k = {:4} * {:.5f} (std. {:.5f})'.format(k, np.mean(evidences), np.std(evidences)))

    # SIVI and IWHVI using independent sampling
    for method_name, tau_force_prior in [('IWHVI', False), ('SIVI-like', True)]:
        for test_iwhvi_samples in args.test_iwhvi_samples:
            x_batch_size = args.max_test_batch_size // (test_iwhvi_samples + 1) + 1
            print('Evaluating evidence, KLs and q gap with {} on {} with M={}, K={}, batch_size={}'.format(
                method_name, args.evaluate_split, args.test_iwae_samples, test_iwhvi_samples, x_batch_size))
            evidences, q_gaps, kl_lower_bounds, kl_upper_bounds = hierarchical_vae.utils.batched_calculate_evidence_q_gap_kls(
                sess, data, vae, args.test_iwae_samples, test_iwhvi_samples, x_batch_size,
                args.n_repeats, tau_force_prior, args.test_iwae_batch_size, tqdm_desc='Calculating')

            eval_data[method_name][test_iwhvi_samples]['evidences'][:] = evidences
            eval_data[method_name][test_iwhvi_samples]['q_gaps'][:] = q_gaps
            eval_data[method_name][test_iwhvi_samples]['kl_upper_bounds'][:] = kl_upper_bounds
            eval_data[method_name][test_iwhvi_samples]['kl_lower_bounds'][:] = kl_lower_bounds

        print('*' * 80)
        print('* {} evidence (using {} bound)'.format(args.evaluate_split, method_name))
        for k in args.test_iwhvi_samples:
            evidences = eval_data[method_name][k]['evidences']
            print('* k = {:4} * {:.5f} (std. {:.5f})'.format(k, np.mean(evidences), np.std(evidences)))

        print('*' * 80)
        print('* {} KL bounds (using {} bound)'.format(args.evaluate_split, method_name))
        for k in args.test_iwhvi_samples:
            kl_lower_bounds = eval_data[method_name][k]['kl_lower_bounds']
            kl_upper_bounds = eval_data[method_name][k]['kl_upper_bounds']
            print('* k = {:4} * {:.5f} (std. {:.5f}) <= KL(q(z)||p(z)) <= {:.5f} (std. {:.5f})'.format(
                k, np.mean(kl_lower_bounds), np.std(kl_lower_bounds), np.mean(kl_upper_bounds), np.std(kl_upper_bounds)
            ))

        print('*' * 80)
        print('* {} log q(z) gap (using {} bound)'.format(args.evaluate_split, method_name))
        for k in args.test_iwhvi_samples:
            q_gaps = eval_data[method_name][k]['q_gaps']
            print('* k = {:4} * log q(z) gap is {:.5f} (std.: {:.5f})'.format(k, np.mean(q_gaps), np.std(q_gaps)))
        print('*' * 80)
        print()

    print('Evaluating KL(tau||q) on {} with M={}'.format(args.evaluate_split, args.test_iwae_samples))
    kls_tau_q = hierarchical_vae.utils.calculate_kl_tau_q(
        sess, data, vae, args.test_iwae_samples, args.diagnostic_kl_batch_size, args.n_repeats,
        tqdm_desc='Calculating KL(tau(psi)||q(psi))')

    print('* The final KL(tau(psi)||q(psi)) on {}: {:.5f} (std.: {:.5f})'.format(
        args.evaluate_split, np.mean(kls_tau_q), np.std(kls_tau_q)))
    print('*' * 80)

    if args.dump_eval_data_path is not None:
        with open(args.dump_eval_data_path, 'wb') as fh:
            pickle.dump(eval_data, fh)


if __name__ == "__main__":
    main()
