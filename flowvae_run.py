import argparse
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf

import flow_vae
import utils
import datasets


def main():
    datasets_available = [f[4:] for f in dir(datasets) if f.startswith('get_') and callable(getattr(datasets, f))]

    argparser = argparse.ArgumentParser(allow_abbrev=False)
    argparser.add_argument('--dataset', choices=datasets_available, default='dynamic_mnist')
    argparser.add_argument('--datasets_dir', default='./datasets/')
    argparser.add_argument('--no_extra_steps', type=int, default=0)
    argparser.add_argument('--q_extra_steps', type=int, default=0)

    argparser.add_argument('--learning_rate', type=float, default=1e-3)
    argparser.add_argument('--annealing_factor', type=float, default=1.0)
    argparser.add_argument('--annealing_speed', type=int, default=100)
    argparser.add_argument('--kl_warm_up_over', type=int, default=1)

    argparser.add_argument('--epochs', type=int, default=3000)
    # StepSchedules are in format [(step_to_begin_at, value_from_now_on_until_next_step)]
    argparser.add_argument('--train_learning_rate_step_schedule', nargs='*',
                           type=utils.make_typed_tuple(int, float), default=[])
    argparser.add_argument('--train_iwae_samples_step_schedule', nargs='*',
                           type=utils.make_typed_tuple(int, int), default=[])
    argparser.add_argument('--train_batch_size', type=int, default=256)
    argparser.add_argument('--train_attempts', type=int, default=10)

    argparser.add_argument('--val_iwae_samples', type=int, default=1000)
    argparser.add_argument('--val_batch_size', type=int, default=50 * 1000)

    argparser.add_argument('--evaluate_every', type=int, default=25)
    argparser.add_argument('--save_every', type=int, default=50)
    argparser.add_argument('--save_path', default=None)

    flow_vae.utils.add_model_args(argparser)
    args = argparser.parse_args()

    dataset = getattr(datasets, 'get_%s' % args.dataset)(args.datasets_dir)
    train_data = dataset.train
    val_data = dataset.validation

    dat_train = []
    dat_val = []

    print('Aguments:')
    for param_name, param_value in sorted(vars(args).items()):
        print('--{:30}: {}'.format(param_name, param_value))
    print('\n')

    print('{} dataset details:'.format(args.dataset))
    for split_name in ['train', 'test', 'validation']:
        print('{}: {}'.format(split_name, getattr(dataset, split_name).num_examples))
    print('\n')

    sess = tf.InteractiveSession()
    vae = flow_vae.utils.get_model(args)

    q_gradients = vae.build_iwhvi_gradients(scope='encoder/')
    all_gradients = vae.build_iwhvi_gradients()

    lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    main_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = main_optimizer.apply_gradients(all_gradients)
    q_finetune_op = main_optimizer.apply_gradients(q_gradients)

    train_iwae_schedule = utils.StepSchedule(args.train_iwae_samples_step_schedule, default=1)
    train_lr_schedule = utils.StepSchedule(args.train_learning_rate_step_schedule, default=args.learning_rate)

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
                np_kl_coef = np.min([epoch / args.kl_warm_up_over, 1])
                iwae_samples = train_iwae_schedule.at(epoch)

                for train_batch in utils.batched_dataset(train_data, args.train_batch_size):
                    binarized_train_batch_x, binarized_train_batch_y = utils.binarize_batch(train_batch)
                    sess.run(train_op, {
                        vae.input_x: binarized_train_batch_x,
                        vae.output_y: binarized_train_batch_y,
                        vae.m_iwae_samples: iwae_samples,
                        vae.kl_coef: np_kl_coef,
                        lr: np_lr
                    })

                    if epoch >= args.no_extra_steps:
                        for _ in range(args.q_extra_steps):
                            sess.run(q_finetune_op, {
                                vae.input_x: binarized_train_batch_x,
                                vae.output_y: binarized_train_batch_y,
                                vae.m_iwae_samples: iwae_samples,
                                vae.kl_coef: np_kl_coef,
                                lr: np_lr
                            })

                if epoch % (5 if epoch < 50 else args.evaluate_every) == 0:
                    avg_evidence_train, = flow_vae.utils.calculate_evidence(
                        sess, train_data, vae, iwae_samples, args.train_batch_size, n_repeats=1)
                    avg_evidence_val, = flow_vae.utils.calculate_evidence(
                        sess, val_data, vae, iwae_samples, args.val_batch_size, n_repeats=1)

                    kl_q, = flow_vae.utils.calculate_kl_q(
                        sess, val_data, vae, args.val_iwae_samples, batch_size=100, n_repeats=1)

                    dat_train.append([epoch, avg_evidence_train])
                    dat_val.append([epoch, avg_evidence_val])

                    utils.print_over("Epoch: {:4}, train_evidence: {:.5f}, "
                                     "val_evidence: {:.5f}, KL(q(z)||p(z)): {:.5f}".format(
                        epoch, avg_evidence_train, avg_evidence_val, kl_q))

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
    avg_evi_val, = flow_vae.utils.calculate_evidence(sess, val_data, vae, args.val_iwae_samples, batch_size_x=1,
                                                     n_repeats=1, tqdm_desc='Calculating val log-evidence')

    print('*' * 30)
    print('* The final model log-evidence on validation is', avg_evi_val)
    print('*' * 30)

    if args.save_path is not None:
        saver.restore(sess, os.path.join(args.save_path, 'model-weights-best_val'))
        avg_evi_val, = flow_vae.utils.calculate_evidence(sess, val_data, vae, args.val_iwae_samples, batch_size_x=1,
                                                         n_repeats=1, tqdm_desc='Calculating val log-evidence')

        print('* The best_val model evidence on validation is', avg_evi_val)
        print('*' * 30)

    if args.save_path is not None:
        dat0 = np.array(dat_train)
        dat1 = np.array(dat_val)

        df0 = pd.DataFrame({'epoch': dat0[:, 0], 'train': dat0[:, 1]})
        df1 = pd.DataFrame({'epoch': dat1[:, 0], 'val': dat1[:, 1]})

        df = pd.concat([df0, df1], ignore_index=True, axis=1)
        df.to_csv(os.path.join(args.save_path, 'data.csv'), index=False)


if __name__ == "__main__":
    main()
