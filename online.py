#!/usr/bin/env python
# =============================================================================
# Copyright (c) 2018 Mengye Ren
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
"""Online experiment.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./online.py    [--dataset {DATASET}]                   \
               [--num_meta_steps {NUM_META_STEPS}]     \
               [--steps_per_update {STEPS_PER_UPDATE}]

Flags:
--dataset:          String. Name of the dataset. Options: `mnist`, `cifar-10`, default `mnist`.
--num_meta_steps:   Int. Number of meta optimization steps every meta update, default 100.
--steps_per_update: Int. Number of steps every meta update, default 10.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle as pkl
import six
import tensorflow as tf

from collections import namedtuple
from matplotlib import pyplot as plt
from tqdm import tqdm

from checkpoint import build_checkpoint
from checkpoint import read_checkpoint
from checkpoint import write_checkpoint
from get_dataset import get_dataset
from logger import get as get_logger
from look_ahead import look_ahead_grads
from models import get_cifar_cnn_config
from models import get_cifar_cnn_model
from models import get_mnist_mlp_config
from models import get_mnist_mlp_model
from models import mlp
from optimizers import LogOptimizer
from train import meta_step
from train import save_results

log = get_logger()

flags = tf.flags
flags.DEFINE_integer('num_meta_steps', 10, 'Number of meta optimization steps')
flags.DEFINE_integer('steps_per_update', 100,
                     'Number of steps per meta updates')
flags.DEFINE_string('dataset', 'mnist', 'Dataset name')
FLAGS = flags.FLAGS

# --------------------------------------------------------------------
# Constants.

# Training curves.
Results = namedtuple('Results', [
    'step', 'train_xent', 'train_acc', 'test_xent', 'test_acc', 'lr',
    'momentum'
])


def _get_exp_logger(sess, log_folder):
    """Gets a TensorBoard logger."""
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    with tf.name_scope('Summary'):
        writer = tf.summary.FileWriter(log_folder, sess.graph)
        summaries = dict()

    class ExperimentLogger():
        def log(self, niter, name, value):
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=value)
            writer.add_summary(summary, niter)

        def flush(self):
            """Flushes results to disk."""

        def close(self):
            """Closes writer."""
            writer.close()

    return ExperimentLogger()


def online_smd(dataset_name='mnist',
               init_lr=1e-1,
               momentum=0.9,
               num_steps=20000,
               middle_decay=False,
               steps_per_update=10,
               smd=True,
               steps_look_ahead=5,
               num_meta_steps=10,
               steps_per_eval=100,
               batch_size=100,
               meta_lr=1e-2,
               print_step=False,
               effective_lr=True,
               negative_momentum=True,
               optimizer='momentum',
               stochastic=True,
               exp_folder='.'):
    """Train an MLP for MNIST.

    Args:
        dataset_name: String. Name of the dataset.
        init_lr: Float. Initial learning rate, default 0.1.
        momentum: Float. Initial momentum, default 0.9.
        num_steps: Int. Total number of steps, default 20000.
        middle_decay: Whether applying manual learning rate decay to 1e-4 from the middle, default False.
        steps_per_update: Int. Number of steps per update, default 10.
        smd: Bool. Whether run SMD.
        steps_look_ahead: Int. Number of steps to look ahead, default 5.
        num_meta_steps: Int. Number of meta steps, default 10.
        steps_per_eval: Int. Number of training steps per evaluation, default 100.
        batch_size: Int. Mini-batch size, default 100.
        meta_lr: Float. Meta learning rate, default 1e-2.
        print_step: Bool. Whether to print loss during training, default True.
        effective_lr: Bool. Whether to re-parameterize learning rate as lr / (1 - momentum), default True.
        negative_momentum: Bool. Whether to re-parameterize momentum as (1 - momentum), default True.
        optimizer: String. Name of the optimizer. Options: `momentum`, `adam, default `momentum`.
        stochastic: Bool. Whether to do stochastic or deterministic look ahead, default True.

    Returns:
        results: Results tuple object.
    """
    dataset = get_dataset(dataset_name)
    dataset_train = get_dataset(
        dataset_name)  # For evaluate training progress (full epoch).
    dataset_test = get_dataset(
        dataset_name, test=True)  # For evaluate test progress (full epoch).

    if dataset_name == 'mnist':
        input_shape = [None, 28, 28, 1]
    elif dataset_name.startswith('cifar'):
        input_shape = [None, 32, 32, 3]

    x = tf.placeholder(tf.float32, input_shape, name="x")
    y = tf.placeholder(tf.int64, [None], name="y")

    if effective_lr:
        init_lr_ = init_lr / (1.0 - momentum)
    else:
        init_lr_ = init_lr

    if negative_momentum:
        init_mom_ = 1.0 - momentum
    else:
        init_mom_ = momentum
    if dataset_name == 'mnist':
        config = get_mnist_mlp_config(
            init_lr_,
            init_mom_,
            effective_lr=effective_lr,
            negative_momentum=negative_momentum)
    elif dataset_name == 'cifar-10':
        config = get_cifar_cnn_config(
            init_lr_,
            init_mom_,
            effective_lr=effective_lr,
            negative_momentum=negative_momentum)
    else:
        raise NotImplemented
    with tf.name_scope('Train'):
        with tf.variable_scope('Model'):
            if dataset_name == 'mnist':
                m = get_mnist_mlp_model(
                    config, x, y, optimizer=optimizer, training=True)
                model = m
            elif dataset_name == 'cifar-10':
                m = get_cifar_cnn_model(
                    config, x, y, optimizer=optimizer, training=True)
                model = m
    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            if dataset_name == 'mnist':
                mtest = get_mnist_mlp_model(config, x, y, training=False)
            elif dataset_name == 'cifar-10':
                mtest = get_cifar_cnn_model(config, x, y, training=False)

    final_lr = 1e-4
    midpoint = num_steps // 2

    if dataset_name == 'mnist':
        num_train = 60000
        num_test = 10000
    elif dataset_name.startswith('cifar'):
        num_train = 50000
        num_test = 10000

    lr_ = init_lr_
    mom_ = init_mom_
    bsize = batch_size
    steps_per_epoch = num_train // bsize
    steps_test_per_epoch = num_test // bsize

    train_xent_list = []
    train_acc_list = []
    test_xent_list = []
    test_acc_list = []
    lr_list = []
    mom_list = []
    step_list = []
    log.info(
        'Applying decay at midpoint with final learning rate = {:.3e}'.format(
            final_lr))

    if 'momentum' in optimizer:
        mom_name = 'mom'
    elif 'adam' in optimizer:
        mom_name = 'beta1'
    else:
        raise ValueError('Unknown optimizer')
    hp_dict = {'lr': init_lr, mom_name: momentum}
    hp_names = hp_dict.keys()
    hyperparams = dict([(hp_name, model.optimizer.hyperparams[hp_name])
                        for hp_name in hp_names])
    grads = model.optimizer.grads
    accumulators = model.optimizer.accumulators
    new_accumulators = model.optimizer.new_accumulators
    loss = model.cost

    # Build look ahead graph.
    look_ahead_ops, hp_grad_ops, zero_out_ops = look_ahead_grads(
        hyperparams, grads, accumulators, new_accumulators, loss)

    # Meta optimizer, use Adam on the log space.
    meta_opt = LogOptimizer(tf.train.AdamOptimizer(meta_lr))
    hp = [model.optimizer.hyperparams[hp_name] for hp_name in hp_names]
    hp_grads_dict = {
        'lr': tf.placeholder(tf.float32, [], name='lr_grad'),
        mom_name: tf.placeholder(
            tf.float32, [], name='{}_grad'.format(mom_name))
    }
    hp_grads_plh = [hp_grads_dict[hp_name] for hp_name in hp_names]
    hp_grads_and_vars = list(zip(hp_grads_plh, hp))
    cgrad = {'lr': (-1e1, 1e1), mom_name: (-1e1, 1e1)}
    cval = {'lr': (1e-4, 1e1), mom_name: (1e-4, 1e0)}
    cgrad_ = [cgrad[hp_name] for hp_name in hp_names]
    cval_ = [cval[hp_name] for hp_name in hp_names]
    meta_train_op = meta_opt.apply_gradients(
        hp_grads_and_vars, clip_gradients=cgrad_, clip_values=cval_)

    var_list = tf.global_variables()
    ckpt = build_checkpoint(tf.global_variables())
    write_op = write_checkpoint(ckpt, var_list)
    read_op = read_checkpoint(ckpt, var_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        exp_logger = _get_exp_logger(sess, exp_folder)

        def log_hp(hp_dict):
            lr_ = hp_dict['lr']
            mom_ = hp_dict['mom']
            # Log current learning rate and momentum.
            if negative_momentum:
                exp_logger.log(ii, 'mom', 1.0 - mom_)
                exp_logger.log(ii, 'log neg mom', np.log10(mom_))
                mom__ = 1.0 - mom_
            else:
                exp_logger.log(ii, 'mom', mom_)
                exp_logger.log(ii, 'log neg mom', np.log10(1.0 - mom_))
                mom__ = mom_

            if effective_lr:
                lr__ = lr_ * (1.0 - mom__)
                eflr_ = lr_
            else:
                lr__ = lr_
                eflr_ = lr_ / (1.0 - mom__)
            exp_logger.log(ii, 'eff lr', eflr_)
            exp_logger.log(ii, 'log eff lr', np.log10(eflr_))
            exp_logger.log(ii, 'lr', lr__)
            exp_logger.log(ii, 'log lr', np.log10(lr__))
            exp_logger.flush()
            return lr__, mom__

        # Assign initial learning rate and momentum.
        m.optimizer.assign_hyperparam(sess, 'lr', lr_)
        m.optimizer.assign_hyperparam(sess, 'mom', mom_)
        train_iter = six.moves.xrange(num_steps)
        if not print_step:
            train_iter = tqdm(train_iter, ncols=0)
        for ii in train_iter:
            # Meta-optimization loop.
            if ii == 0 or ii % steps_per_update == 0:
                if ii < midpoint and smd:
                    if stochastic:
                        data_list = [
                            dataset.next_batch(bsize)
                            for step in six.moves.xrange(steps_look_ahead)
                        ]
                        # Take next few batches for last step evaluation.
                        eval_data_list = [
                            dataset.next_batch(bsize)
                            for step in six.moves.xrange(steps_look_ahead)
                        ]
                    else:
                        data_entry = dataset.next_batch(bsize)
                        data_list = [data_entry] * steps_look_ahead
                        # Use the deterministic batch for last step evaluation.
                        eval_data_list = [data_list[0]]
                    sess.run(write_op)
                    for ms in six.moves.xrange(num_meta_steps):
                        cost, hp_dict = meta_step(sess, model, data_list,
                                                  look_ahead_ops, hp_grad_ops,
                                                  hp_grads_plh, meta_train_op,
                                                  eval_data_list)
                        sess.run(read_op)
                        for hpname, hpval in hp_dict.items():
                            model.optimizer.assign_hyperparam(
                                sess, hpname, hpval)
                    lr_ = hp_dict['lr']
                    mom_ = hp_dict['mom']
                else:
                    hp_dict = sess.run(model.optimizer.hyperparams)
                lr_log, mom_log = log_hp(hp_dict)
                lr_list.append(lr_log)
                mom_list.append(mom_log)

            if ii == midpoint:
                lr_before_mid = hp_dict['lr']
                tau = (num_steps - midpoint) / np.log(lr_before_mid / final_lr)

            if ii > midpoint:
                lr_ = np.exp(-(ii - midpoint) / tau) * lr_before_mid
                m.optimizer.assign_hyperparam(sess, 'lr', lr_)

            # Run regular training.
            if lr_ > 1e-6:
                xd, yd = dataset.next_batch(bsize)
                cost_, _ = sess.run(
                    [m.cost, m.train_op], feed_dict={
                        m.x: xd,
                        m.y: yd
                    })

            # Evaluate every certain number of steps.
            if ii == 0 or (ii + 1) % steps_per_eval == 0:
                test_acc = 0.0
                test_xent = 0.0
                train_acc = 0.0
                train_xent = 0.0

                # Report full epoch training loss.
                for jj in six.moves.xrange(steps_per_epoch):
                    xd, yd = dataset_train.next_batch(bsize)
                    xent_, acc_ = sess.run(
                        [m.cost, m.acc], feed_dict={
                            x: xd,
                            y: yd
                        })
                    train_xent += xent_ / float(steps_per_epoch)
                    train_acc += acc_ / float(steps_per_epoch)
                step_list.append(ii + 1)
                train_xent_list.append(train_xent)
                train_acc_list.append(train_acc)
                dataset_train.reset()

                # Report full epoch validation loss.
                for jj in six.moves.xrange(steps_test_per_epoch):
                    xd, yd = dataset_test.next_batch(bsize)
                    xent_, acc_ = sess.run(
                        [mtest.cost, mtest.acc], feed_dict={
                            x: xd,
                            y: yd
                        })
                    test_xent += xent_ / float(steps_test_per_epoch)
                    test_acc += acc_ / float(steps_test_per_epoch)
                test_xent_list.append(test_xent)
                test_acc_list.append(test_acc)
                dataset_test.reset()

                # Log training progress.
                exp_logger.log(ii, 'train loss', train_xent)
                exp_logger.log(ii, 'log train loss', np.log10(train_xent))
                exp_logger.log(ii, 'test loss', test_xent)
                exp_logger.log(ii, 'log test loss', np.log10(test_xent))
                exp_logger.log(ii, 'train acc', train_acc)
                exp_logger.log(ii, 'test acc', test_acc)
                exp_logger.flush()

                if print_step:
                    log.info((
                        'Steps {:d} T Xent {:.3e} T Acc {:.3f} V Xent {:.3e} V Acc {:.3f} '
                        'LR {:.3e}').format(ii + 1, train_xent,
                                            train_acc * 100.0, test_xent,
                                            test_acc * 100.0, lr_))

    return Results(
        step=np.array(step_list),
        train_xent=np.array(train_xent_list),
        train_acc=np.array(train_acc_list),
        test_xent=np.array(test_xent_list),
        test_acc=np.array(test_acc_list),
        lr=np.array(lr_list),
        momentum=np.array(mom_list))


def plot_report_figure_combined(steps,
                                values1,
                                values2,
                                condition,
                                title,
                                ylabel1,
                                ylabel2,
                                filename,
                                subsample=1,
                                figsize=(8, 7),
                                include_legend=True,
                                top_left=None,
                                ylim1=None,
                                ylim2=None):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    ax = axes.flatten()
    ax1 = ax[0]
    ax2 = ax[1]

    num_steps = values1.shape[0]
    num_exp = values1.shape[1]
    values1 = values1.reshape([-1, subsample, num_exp])
    values1 = np.mean(values1, axis=1)
    values1 = np.expand_dims(values1, 0)

    def empty_fmt(x, pos):
        return ''

    empty_formatter = ticker.FuncFormatter(empty_fmt)
    lns = []
    color_list = ['red'] + ['blue'] * (num_exp - 1)
    for ii in range(num_exp):
        values_ = values1[0, :, ii]
        color_ = color_list[ii]
        lns.append(ax1.plot(steps, values_, color=color_, linewidth=2)[0])
    ax1.grid(color='k', linestyle=':', linewidth=1)
    ax1.set_yscale('log')
    if ylim1 is not None:
        ax1.set_ylim(*ylim1)
    ax1.tick_params(labelsize=18)
    ax1.xaxis.set_major_formatter(empty_formatter)
    ax1.legend(labels=condition, handles=lns[:2], loc=3)
    plt.setp(ax1.get_legend().get_texts(), fontsize=18)
    ax1.set_title(title, fontsize=30)
    ax1.set_ylabel(ylabel1, fontsize=24)

    ax2.xaxis.major.formatter._useMathText = True
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=18)
    values2 = values2.reshape([-1, subsample, num_exp])
    values2 = values2[:, 0, :]
    for ii in range(num_exp):
        values_ = values2[:, ii]
        color_ = color_list[ii]
        ax2.plot(steps, values_, color=color_, linewidth=2)
    ax2.grid(color='k', linestyle=':', linewidth=1)
    ax2.set_xlabel("Steps", fontsize=24)
    ax2.set_ylabel(ylabel2, fontsize=24)
    plt.tight_layout(pad=2.0)
    plt.savefig(filename)


def open_folder(folder, stochastic=True):
    """Reads experiment results from a bunch of pklz files.

    Args:
        folder: String. Path to the experiment folder.

    Returns:
        configs: A list of experimental configurations.
        results: A list of experimental results.
    """
    if not os.path.exists(folder):
        raise ValueError("{} not found".format(folder))
    print(folder)
    files = list(
        sorted(filter(lambda x: x.startswith('0'), os.listdir(folder))))
    print(files)
    if stochastic:
        files = list(filter(lambda x: 'stoc' in x or 'manual' in x, files))
    else:
        files = list(filter(lambda x: 'det' in x or 'manual' in x, files))
    files = list(map(lambda x: os.path.join(x, 'result.npy'), files))
    results_list = []
    print(files)
    for fname in files:
        results_list.append(
            np.load(os.path.join(folder, fname), encoding='latin1'))
    return results_list


def plot_folder(folder, title, stochastic=True):
    """Plots an experiment folder.

    Args:
        folder: String. Path to the folder.
        show: Bool. Whether to show the plots in a window.
    """
    results = open_folder(folder, stochastic=stochastic)
    assert len(results) > 0, 'Cannot find any results'
    ce_values = np.concatenate(
        [np.expand_dims(r.item()['train_xent'][:-1], 1) for r in results],
        axis=1)
    steps = np.concatenate(
        [np.expand_dims(r.item()['step'][:-1], 1) for r in results], axis=1)
    alpha_eff_values = np.concatenate(
        [
            np.expand_dims(r.item()['lr'] / (1.0 - r.item()['momentum']), 1)
            for r in results
        ],
        axis=1)
    condition = ['Manual', 'SMD']
    plot_report_figure_combined(
        steps,
        ce_values,
        alpha_eff_values,
        condition,
        title,
        'Loss',
        'Eff. Learning Rate',
        os.path.join(
            folder, 'combined_{}.pdf'.format('stoc' if stochastic else 'det')),
        include_legend=True)


def run_dataset(exp_folder, dataset_name, best_lr, lr_list, optimizer,
                num_steps, num_meta_steps, steps_per_update):
    """Run online experiments for a dataset.

    Args:
        dataset_name: String. Name of the dataset.
        best_lr: Float. Best LR for the dataset.
        lr_list: List of initial learning rate to try from.
        num_steps: Int. Number of total training step.
        num_meta_steps: Int. Number of meta optimization steps per update.
        steps_per_update: Int. Number of regular training steps per update.
    """
    with tf.Graph().as_default():
        result_folder = os.path.join(exp_folder, '000_manual_best')
        savepath = os.path.join(result_folder, 'result.npy')
        if os.path.exists(savepath):
            log.info('{} exists, skip'.format(savepath))
        else:
            save_results(savepath,
                         online_smd(
                             dataset_name=dataset_name,
                             init_lr=best_lr,
                             steps_per_update=steps_per_update,
                             smd=False,
                             optimizer=optimizer,
                             num_steps=num_steps,
                             exp_folder=result_folder))

    id_len = 1
    for jj, init_lr in enumerate(lr_list):
        with tf.Graph().as_default():
            result_folder = os.path.join(exp_folder,
                                         '{:03d}_stoc_lr_{:.0e}'.format(
                                             jj + id_len, init_lr))
            savepath = os.path.join(result_folder, 'result.npy')
            if os.path.exists(savepath):
                log.info('{} exists, skip'.format(savepath))
            else:
                save_results(savepath,
                             online_smd(
                                 dataset_name=dataset_name,
                                 init_lr=init_lr,
                                 num_meta_steps=num_meta_steps,
                                 steps_per_update=steps_per_update,
                                 stochastic=True,
                                 optimizer=optimizer,
                                 num_steps=num_steps,
                                 exp_folder=result_folder))

    id_len = len(lr_list) + 1
    for jj, init_lr in enumerate(lr_list):
        with tf.Graph().as_default():
            result_folder = os.path.join(exp_folder,
                                         '{:03d}_det_lr_{:.0e}'.format(
                                             jj + id_len, init_lr))
            savepath = os.path.join(result_folder, 'result.npy')
            if os.path.exists(savepath):
                log.info('{} exists, skip'.format(savepath))
            else:
                save_results(savepath,
                             online_smd(
                                 dataset_name=dataset_name,
                                 init_lr=init_lr,
                                 num_meta_steps=num_meta_steps,
                                 steps_per_update=steps_per_update,
                                 stochastic=False,
                                 optimizer=optimizer,
                                 num_steps=num_steps,
                                 exp_folder=result_folder))


def main():
    optimizer = 'momentum'
    exp_folder = os.path.join('results', FLAGS.dataset, 'online', optimizer)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    if FLAGS.dataset == 'mnist':
        run_dataset(exp_folder, 'mnist', 1e-1, [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
                    optimizer, 50000, FLAGS.num_meta_steps,
                    FLAGS.steps_per_update)
    elif FLAGS.dataset == 'cifar-10':
        run_dataset(exp_folder, 'cifar-10', 5e-3, [5e-4, 1e-3, 5e-3, 1e-2],
                    optimizer, 50000, FLAGS.num_meta_steps,
                    FLAGS.steps_per_update)
    else:
        raise ValueError('Dataset not supported.')
    plot_folder(exp_folder, FLAGS.dataset.upper(), True)
    plot_folder(exp_folder, FLAGS.dataset.upper(), False)


if __name__ == '__main__':
    main()
