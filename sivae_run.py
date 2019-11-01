import argparse
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf

import hierarchical_vae
import utils
import datasets


def main():
    datasets_available = [f[4:] for f in dir(datasets) if f.startswith('get_') and callable(getattr(datasets, f))]

    argparser = argparse.ArgumentParser(allow_abbrev=False)
    argparser.add_argument('--dataset', choices=datasets_available, default='dynamic_mnist')
    argparser.add_argument('--datasets_dir', default='./datasets/')

    argparser.add_argument('--learning_rate', type=float, default=1e-3)
    argparser.add_argument('--annealing_factor', type=float, default=1.0)
    argparser.add_argument('--annealing_speed', type=int, default=100)
    argparser.add_argument('--q_kl_warm_up_fraction', type=float, default=0.)
    argparser.add_argument('--q_kl_warm_up_cycles', type=int, default=0)

    argparser.add_argument('--epochs', type=int, default=3000)
    # StepSchedules are in format [(step_to_begin_at, value_from_now_on_until_next_step)]
    argparser.add_argument('--train_learning_rate_step_schedule', nargs='*',
                           type=utils.make_typed_tuple(int, float), default=[])
    argparser.add_argument('--train_iwae_samples_step_schedule', nargs='*',
                           type=utils.make_typed_tuple(int, int), default=[])
    argparser.add_argument('--train_sivi_samples_step_schedule', nargs='*',
                           type=utils.make_typed_tuple(int, int), default=[(250, 5), (500, 25), (1000, 50)])
    argparser.add_argument('--train_batch_size', type=int, default=256)
    argparser.add_argument('--train_attempts', type=int, default=10)

    argparser.add_argument('--val_iwae_samples', type=int, default=1000)
    argparser.add_argument('--val_iwhvi_samples', type=int, default=100)
    argparser.add_argument('--val_batch_size', type=int, default=50 * 1000)

    argparser.add_argument('--evaluate_every', type=int, default=25)
    argparser.add_argument('--save_every', type=int, default=50)
    argparser.add_argument('--save_path', default=None)

    hierarchical_vae.utils.add_model_args(argparser)
    args = argparser.parse_args()

    dataset = getattr(datasets, 'get_%s' % args.dataset)(args.datasets_dir)
    train_data = dataset.train
    val_data = dataset.validation

    print('Aguments:')
    for param_name, param_value in sorted(vars(args).items()):
        print('--{:30}: {}'.format(param_name, param_value))
    print('\n')

    print('{} dataset details:'.format(args.dataset))
    for split_name in ['train', 'test', 'validation']:
        print('{}: {}'.format(split_name, getattr(dataset, split_name).num_examples))
    print('\n')

    sess = tf.InteractiveSession()
    vae = hierarchical_vae.utils.get_model(args)

    lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    main_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = main_optimizer.minimize(-vae.reused_sivi_avg_elbo)

    train_iwae_schedule = utils.StepSchedule(args.train_iwae_samples_step_schedule, default=1)
    train_sivi_schedule = utils.StepSchedule(args.train_sivi_samples_step_schedule, default=0)
    train_lr_schedule = utils.StepSchedule(args.train_learning_rate_step_schedule, default=args.learning_rate)

    q_kl_schedule = utils.CyclicTemperatureSchedule(args.epochs, args.q_kl_warm_up_cycles, args.q_kl_warm_up_fraction)

    saver = tf.train.Saver(max_to_keep=100)
    best_val_saver = tf.train.Saver()
    if args.save_path is not None:
        train_writer = tf.summary.FileWriter(args.save_path + '/logs')
        train_writer.add_graph(sess.graph)
        # TODO: add summaries

    for attempt in range(args.train_attempts):
        utils.print_over('Attempt #{}'.format(attempt + 1))

        sess.run(tf.global_variables_initializer())
        best_val_score = -np.inf
        try:
            tqdm_t = tqdm(range(args.epochs), unit='epoch', desc='Training')
            for epoch in tqdm_t:
                np_lr = train_lr_schedule.at(epoch) * args.annealing_factor ** (epoch / args.annealing_speed)
                np_kl_coef = q_kl_schedule.at(epoch)

                for train_batch in utils.batched_dataset(train_data, args.train_batch_size):
                    binarized_train_batch_x, binarized_train_batch_y = utils.binarize_batch(train_batch)
                    sess.run(train_op, {
                        vae.input_x: binarized_train_batch_x,
                        vae.output_y: binarized_train_batch_y,
                        vae.k_iwhvi_samples: train_sivi_schedule.at(epoch),
                        vae.m_iwae_samples: train_iwae_schedule.at(epoch),
                        vae.kl_coef: np_kl_coef,
                        vae.tau_force_prior: True,
                        vae.iwhvi_kl_coef: 1.,
                        lr: np_lr
                    })

                if epoch % (5 if epoch < 50 else args.evaluate_every) == 0:
                    avg_evidence_train, = hierarchical_vae.utils.calculate_evidence(
                        sess, train_data, vae, 1, 100, args.train_batch_size, 1, True)
                    avg_evidence_val, = hierarchical_vae.utils.calculate_evidence(
                        sess, val_data, vae, 1, 100,
                        (args.val_batch_size - 1) // ((100 + 1) * 1) + 1, 1, True)

                    utils.print_over("Epoch: {:4}, train_evidence: {:.5f}, val_evidence: {:.5f}".format(
                        epoch, avg_evidence_train, avg_evidence_val))

                    if avg_evidence_val > best_val_score:
                        best_val_score = avg_evidence_val
                        utils.save_weights(sess, args.save_path, suffix='best_val', saver=best_val_saver)

                    if any(np.isnan([avg_evidence_train, avg_evidence_val])):
                        if epoch < 300:
                            tqdm_t.close()
                            utils.print_over('nans detected, restarting')
                            break
                        else:
                            raise RuntimeError('nans detected')

                if epoch % args.save_every == 0 and epoch:
                    utils.save_weights(sess, args.save_path, suffix='ep%d' % epoch, saver=saver)
            else:
                break
        except KeyboardInterrupt:
            print('Manual stop')
            break

    utils.save_weights(sess, args.save_path, suffix='final', saver=saver)
    avg_evi_val, = hierarchical_vae.utils.calculate_evidence(
        sess, val_data, vae, args.val_iwae_samples, args.val_iwhvi_samples, 1, 1,
        True, tqdm_desc='Calculating val log-evidence')

    print('*' * 30)
    print('* The final model log-evidence on validation is', avg_evi_val)
    print('*' * 30)

    saver.restore(sess, os.path.join(args.save_path, 'model-weights-best_val'))
    avg_evi_val, = hierarchical_vae.utils.calculate_evidence(
        sess, val_data, vae, args.val_iwae_samples, args.val_iwhvi_samples, 1, 1,
        True, tqdm_desc='Calculating val log-evidence')

    print('* The best_val model evidence on validation is', avg_evi_val)
    print('*' * 30)


if __name__ == "__main__":
    main()
