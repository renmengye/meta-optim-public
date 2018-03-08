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
"""Offline experiment.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./offline.py                                                            \
                          --high_log_decay     [UPPER BOUND LOG DECAY]  \
                          --high_log_lr        [UPPER BOUND LOG LR]     \
                          --low_log_decay      [LOWER BOUND LOG DECAY]  \
                          --low_log_lr         [LOWER BOUND LOG LR]     \
                          --time_const         [TIME CONST]             \
                          --num_meta_steps     [NUM META STEPS]         \
                          --num_pretrain_steps [NUM PRETRAIN STEPS]     \
                          --num_samples        [NUM SAMPLES]            \
                          --seed               [RANDOM SEED]            \
                          --init_decay         [INIT DECAY LIST]        \
                          --init_lr            [INIT LR LIST]           \
                          --num_steps          [NUM STEPS]

Flags:
--high_log_decay:      Float. Upper bound of log decay, used for surface search, default 2.0.
--high_log_lr:         Float. Upper bound of log learning rate, used for surface search, default -0.3.
--low_log_decay:       Float. Lower bound of log decay, used for surface search, default -1.0.
--low_log_lr:          Float. Lower bound of log learning rate, used for surface search, default -1.0.
--time_const:          Float. Decay time constant, default 5000.0.
--num_meta_steps:      Int. Number of meta optimization steps, used for SMD, default 5000.
--num_pretrain_steps:  Int. Number of pretrain steps, default 50.
--num_samples:         Int. Number of random samples, used for surface search, default 2000.
--seed:                Int. Random seed, default 0.
--init_decay:          String. Comma delimited list of float values of initial decay, used for SMD, default 0.1,10.0.
--init_lr:             String. Comma delimited list of float values of initial lr, used for SMD, default 0.01.
--num_steps:           String. Comma delimited list of int values of number of look ahead steps, default 100, 1000, 5000, 20000.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import pickle as pkl
import six
import tensorflow as tf

from collections import namedtuple
from matplotlib import pyplot as plt
from tqdm import tqdm

from checkpoint import build_checkpoint, write_checkpoint, read_checkpoint
from get_dataset import get_dataset
from logger import get as get_logger
from look_ahead import look_ahead_grads
from models import mlp, get_mnist_mlp_model, get_mnist_mlp_config
from optimizers import LogOptimizer
from train import train_steps, train_mnist_mlp_with_test, meta_step, save_results

log = get_logger()

flags = tf.flags
flags.DEFINE_float('high_log_decay', 2.3, 'Highest log decay')
flags.DEFINE_float('high_log_lr', -0.3, 'Highest log learning rate')
flags.DEFINE_float('low_log_decay', -2.0, 'Lowest log decacy')
flags.DEFINE_float('low_log_lr', -3.0, 'Lowest log learning rate')
flags.DEFINE_float('meta_lr', 1e-2, 'Meta learning rate')
flags.DEFINE_float('time_const', 5000.0, 'Decay time constant')
flags.DEFINE_integer('num_meta_steps', 5000, 'Number of meta optimization steps')
flags.DEFINE_integer('num_pretrain_steps', 50, 'Number of pretraining steps')
flags.DEFINE_integer('num_samples', 2500, 'Number of random samples')
flags.DEFINE_integer('seed', 0, 'Random seed for samples')
flags.DEFINE_string('init_decay', '0.1,10.0',
                    'Comma delimited string of initial decay exponent for offline SMD')
flags.DEFINE_string('init_lr', '0.01',
                    'Comma delimited string of initial learning rates for offline SMD')
flags.DEFINE_string('num_steps', '100,1000,5000,20000',
                    'Comma delimited string for number of steps to look ahead')
flags.DEFINE_string('run', None, 'Which experiment to run, options `surface`, `smd`, `best`.')

FLAGS = flags.FLAGS

# --------------------------------------------------------------------
# Constants.
# File to store pretrained model.
PRETRAIN_FILE = 'results/mnist/pretrain_{}/pretrain_{}'.format(FLAGS.num_pretrain_steps,
                                                               FLAGS.num_pretrain_steps)
# Batch size for training.
BATCH_SIZE = 100
# Number of training samples.
NUM_TRAIN = 60000
# Number of test samples.
NUM_TEST = 10000
# Fix all momentum = 0.9.
MOMENTUM = 0.9


def run_random_search(num_steps, lr_limit, decay_limit, num_samples, ckpt, output, seed=0):
    """Random search hyperparameters to plot the surface.

    Args:
        num_steps: Int. Number of look ahead steps.
        lr_limit: Tuple. Two float denoting the lower and upper search bound.
        decay_limit: Tuple. Two float denoting the lower and upper search bound.
        num_samples: Int. Number of samples to try.
        ckpt: String. Pretrain checkpoint name.
        output: String. Output CSV results file name.

    Returns:
    """
    bsize = BATCH_SIZE
    log.info('Writing output to {}'.format(output))
    log_folder = os.path.dirname(output)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    with tf.Graph().as_default(), tf.Session() as sess:
        dataset = get_dataset('mnist')
        config = get_mnist_mlp_config(0.0, MOMENTUM)
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
        y = tf.placeholder(tf.int64, [None], name="y")
        with tf.name_scope('Train'):
            with tf.variable_scope('Model'):
                m = get_mnist_mlp_model(config, x, y, training=True)
        var_to_restore = list(filter(lambda x: 'Momentum' not in x.name, tf.global_variables()))
        saver = tf.train.Saver(var_to_restore)
        # 200 points in the learning rate list, and 100 points in the decay list.
        # random sample 1000.
        rnd = np.random.RandomState(seed)
        # Get a list of stochastic batches first.
        data_list = [dataset.next_batch(bsize) for step in six.moves.xrange(num_steps)]
        settings = []
        for run in tqdm(six.moves.xrange(num_samples), ncols=0, desc='{} steps'.format(num_steps)):
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt)
            lr = np.random.rand() * (lr_limit[1] - lr_limit[0]) + lr_limit[0]
            lr = np.exp(lr * np.log(10))
            decay = rnd.uniform(0, 1) * (decay_limit[1] - decay_limit[0]) + decay_limit[0]
            decay = np.exp(decay * np.log(10))
            m.optimizer.assign_hyperparam(sess, 'lr', lr)
            loss, final_loss = train_steps(sess, m, data_list, init_lr=lr, decay_const=decay)
            settings.append([lr, decay, final_loss])
        settings = np.array(settings)
        np.savetxt(output, settings, delimiter=',', header='lr,decay,loss')
        loss = settings[:, 2]
        sort_idx = np.argsort(loss)
        sorted_settings = settings[sort_idx]
        print('======')
        print('Best 10 settings')
        for ii in six.moves.xrange(10):
            aa = sorted_settings[ii, 0]
            decay = sorted_settings[ii, 1]
            loss = sorted_settings[ii, 2]
            print('Alpha', aa, 'Decay', decay, 'Loss', loss)
    return sorted_settings[0, 0], sorted_settings[0, 1], sorted_settings[0, 2]


def run_final_setup(lr_list, decay_list, save_list, num_steps=20000, steps_per_eval=50):
    """Gets the results for the final setup.

    Args:
        lr_list: List of learning rate values.
        decay_list: List of decay values.
        save_list: List of save filenames.
    """
    # Pretrain
    dataset = get_dataset('mnist')
    data_list = [dataset.next_batch(100) for step in six.moves.xrange(num_steps)]
    data_list_eval = data_list[:600]
    dataset_test = get_dataset('mnist', test=True)
    data_list_test = [dataset_test.next_batch(100) for step in six.moves.xrange(100)]

    for ii, (lr, decay, save) in enumerate(zip(lr_list, decay_list, save_list)):
        print('-' * 80)
        log.info('Running lr = {:.3e} decay = {:.3e}'.format(lr, decay))
        with tf.Graph().as_default():
            results = train_mnist_mlp_with_test(
                init_lr=lr,
                num_steps=num_steps,
                decay_const=decay,
                steps_per_eval=steps_per_eval,
                inverse_decay=True,
                pretrain_ckpt=PRETRAIN_FILE,
                print_step=False,
                data_list=data_list,
                data_list_eval=data_list_eval,
                data_list_test=data_list_test)
        log.info('Final Train Cost {:.3e} Train Acc {:.3f} Test Cost {:.3e} Test Acc {:.3f}'.format(
            results.train_xent[-1], results.train_acc[-1], results.test_xent[-1],
            results.test_acc[-1]))
        print(results.train_xent)
        print('Train Cost', results.train_xent[-1])
        save_results(save, results)


def plot_final_setup(output_list, lr_list, decay_list, step_list, output_fname):
    """Plots training curve for the final chosen settings.

    Args:
        output_list:
        lr_list:
        decay_list:
        step_list:
        output_fname:
    """
    # Plot training loss.
    fig, axes = plt.subplots(3, 1, figsize=(11, 14))
    color_list = ['r', 'b', 'c', 'k']
    axes = axes.flatten()
    for ii, (fname, lr, decay) in enumerate(zip(output_list, lr_list, decay_list)):
        print(fname)
        # results = pkl.load(open(fname, 'rb'))
        results = np.load(fname)
        results = results.item()
        axes[0].plot(results['step'], results['train_xent'], color=color_list[ii], linewidth=3)
        test_line, = axes[1].plot(
            results['step'], (1.0 - results['test_acc']) * 100.0, color=color_list[ii], linewidth=3)
        train_line, = axes[1].plot(
            results['step'], (1.0 - results['train_acc']) * 100.0,
            color=color_list[ii],
            linestyle=":",
            linewidth=3)
        axes[2].plot(results['step'], results['lr'], color=color_list[ii], linewidth=3)

    labelsize = 24
    axes[0].legend(['{} Steps'.format(step) for step in step_list])
    axes[0].set_ylabel('Train Loss', fontsize=labelsize)
    axes[0].set_xlabel('Step', fontsize=labelsize)
    axes[0].set_xticks([0, 5000, 10000, 15000, 20000])
    axes[0].xaxis.major.formatter._useMathText = True
    #axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axes[0].set_yscale('log')
    axes[0].tick_params(labelsize=24)

    axes[1].legend(labels=['Train', 'Test'], handles=[train_line, test_line])
    axes[1].set_ylabel('Error (%)', fontsize=labelsize)
    axes[1].set_xlabel('Step', fontsize=labelsize)
    axes[1].set_xticks([0, 5000, 10000, 15000, 20000])
    axes[1].xaxis.major.formatter._useMathText = True
    #axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axes[1].tick_params(labelsize=24)

    axes[2].set_ylim(1e-3, 1e0)
    axes[2].set_ylabel('Learning Rate', fontsize=labelsize)
    axes[2].set_xlabel('Step', fontsize=labelsize)
    axes[2].set_xticks([0, 5000, 10000, 15000, 20000])
    axes[2].xaxis.major.formatter._useMathText = True
    #axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axes[2].tick_params(labelsize=24)
    axes[2].set_yscale('log')

    plt.setp(axes[0].get_legend().get_texts(), fontsize=24)
    plt.setp(axes[1].get_legend().get_texts(), fontsize=24)
    plt.tight_layout()
    plt.savefig(output_fname)


def prepare_surface(fname, alpha_limit=None, decay_limit=None, imsize=50):
    """Preprocess surface data."""
    data = np.loadtxt(fname, skiprows=1, delimiter=',')
    data = data[np.logical_not(np.isnan(data[:, 2]))]
    # data[:, 0] = data[:, 0] * 10.0  # Make it effective.

    if alpha_limit is None:
        min_alpha = data[:, 0].min()
        max_alpha = data[:, 0].max()
        alpha_limit = (min_alpha, max_alpha)
    else:
        min_alpha = alpha_limit[0]
        max_alpha = alpha_limit[1]

    min_log_alpha = np.log10(min_alpha)
    max_log_alpha = np.log10(max_alpha)

    if decay_limit is None:
        min_decay = data[:, 1].min()
        max_decay = data[:, 1].max()
        decay_limit = (min_decay, max_decay)
    else:
        min_decay = decay_limit[0]
        max_decay = decay_limit[1]

    min_log_decay = np.log10(min_decay)
    max_log_decay = np.log10(max_decay)

    grid_log_alpha = np.linspace(min_log_alpha, max_log_alpha, imsize)
    grid_log_decay = np.linspace(min_log_decay, max_log_decay, imsize)

    # Use Gaussian interpolation.
    # 1st dim -> alpha
    # 2nd dim -> decay
    delta_log_decay = (max_log_decay - min_log_decay) / imsize
    delta_log_alpha = (max_log_alpha - min_log_alpha) / imsize
    sigma = np.sqrt(delta_log_decay**2 + delta_log_alpha**2)
    dist_a = 0.5 * (grid_log_alpha.reshape([1, -1]) - np.log10(data[:, 0:1]))**2 / sigma**2
    dist_d = 0.5 * (grid_log_decay.reshape([1, -1]) - np.log10(data[:, 1:2]))**2 / sigma**2
    dist = np.expand_dims(dist_a, axis=2)**2 + np.expand_dims(dist_d, axis=1)**2
    prob = np.exp(-dist)
    z = prob / prob.sum(axis=0, keepdims=True)
    value = (z * np.log10(data[:, 2]).reshape([-1, 1, 1])).sum(axis=0)
    return value, alpha_limit, decay_limit, grid_log_alpha, grid_log_decay, data


def plot_surface(fname_list, step_list):
    """Plot multiple hyperparameter surfaces.

    Args:
        fname_list: List of filenames.
        step_list: List of look ahead steps.

    Returns:
        axes:
        best_alpha:
        best_decay:
        alpha_limit
    """
    imsize = 50
    fig, axes = plt.subplots(nrows=1, ncols=len(fname_list), figsize=(6 * len(step_list), 6))
    alpha_limit = None
    decay_limit = None
    losses = [np.loadtxt(ff, skiprows=1, delimiter=',')[:, 2] for ff in fname_list]
    losses = np.concatenate(losses, axis=0)
    max_loss, min_loss = np.log10(losses.max()), np.log10(losses.min())
    best_alpha = []
    best_decay = []
    if len(fname_list) == 1:
        axes = [axes]
    else:
        axes = axes.flat
    for fname, ax, t in zip(fname_list, axes, step_list):
        value, _alpha_limit, _decay_limit, grid_log_alpha, grid_log_decay, data = prepare_surface(
            fname, imsize=imsize, alpha_limit=alpha_limit, decay_limit=decay_limit)

        # Take an average of the top 50 hyperparameter settings.
        top = 50
        idx = np.argsort(value.reshape([-1]))
        ba = 0.0
        bd = 0.0
        # print(idx)
        for idx_ in idx[:top]:
            row = idx_ // imsize
            col = idx_ % imsize
            # print(row, col)
            ba += grid_log_alpha[row]
            bd += grid_log_decay[col]
        best_alpha.append(np.exp(np.log(10) * ba / float(top)))
        best_decay.append(np.exp(np.log(10) * bd / float(top)))

        if alpha_limit is None:
            alpha_limit = _alpha_limit
        if decay_limit is None:
            decay_limit = _decay_limit
        _y = np.linspace(np.log10(alpha_limit[0]), np.log10(alpha_limit[1]), imsize)
        _x = np.linspace(np.log10(decay_limit[0]), np.log10(decay_limit[1]), imsize)

        X, Y = np.meshgrid(_x, _y)
        origin = 'lower'
        im = ax.contourf(
            np.exp(X * np.log(10)), np.exp(Y * np.log(10)), value, 12, cmap='gray', origin=origin)
        im = ax.contour(im, levels=im.levels, colors='c', origin=origin)
        if plt.rcParams["text.usetex"]:
            fmt = r'%r \%%'
        else:
            fmt = '%r %%'
        ax.clabel(im, im.levels, inline=True)
        ax.set_ylabel("Initial Learning Rate", fontsize=20)
        ax.set_xticks(_x[::10], ["10^{:.0e}".format(dd) for dd in grid_log_decay[::10]])
        ax.set_xlabel("Decay Exponent", fontsize=20)
        ax.set_title("{} steps".format(t), fontsize=24)
        ax.set_xscale('log')
        ax.set_yscale('log')
        zed = [tick.label.set_fontsize(18) for tick in ax.get_yaxis().get_major_ticks()]
        zed = [tick.label.set_fontsize(18) for tick in ax.get_xaxis().get_major_ticks()]
        if t > step_list[0]:
            ax.yaxis.set_visible(False)
    plt.tight_layout(h_pad=1.0, w_pad=3.0)
    return axes, best_alpha, best_decay


def pretrain(num_pretrain_steps):
    """Pretrain the network."""
    log.info('Pretraining for {:d} steps'.format(num_pretrain_steps))
    pretrain_dir = os.path.dirname(PRETRAIN_FILE)
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)
    if not os.path.exists(PRETRAIN_FILE + '.meta'):
        with tf.Graph().as_default():
            train_mnist_mlp_with_test(
                num_steps=num_pretrain_steps,
                middle_decay=False,
                inverse_decay=False,
                save_ckpt=PRETRAIN_FILE,
                print_step=True)
    else:
        log.info('Pretrain file already exists')


def _get_run_number(log_folder):
    folders = os.listdir(log_folder)
    print(folders)
    folders = filter(lambda x: x.startswith('run'), folders)
    runname = [int(ff[3:]) for ff in folders]
    if len(runname) > 0:
        return max(runname) + 1
    else:
        return 0


def _get_exp_logger(sess, log_folder):
    """Gets a TensorBoard logger."""
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


def run_offline_smd(num_steps,
                    init_lr,
                    init_decay,
                    meta_lr,
                    num_meta_steps,
                    momentum=MOMENTUM,
                    effective_lr=False,
                    negative_momentum=False,
                    pretrain_ckpt=None,
                    output_fname=None,
                    seed=0):
    """Run offline SMD experiments.

    Args:
        init_lr: Initial learning rate.
        init_decay: Initial decay constant.
        data_list: List of tuples of inputs and labels.
        meta_lr: Float. Meta descent learning rate.
        num_meta_steps: Int. Number of meta descent steps.
        momentum: Float. Momentum.
        effective_lr: Bool. Whether to optimize in the effective LR space.
        negative_momentum: Bool. Whether to optimize in the negative momentum space.
    """
    bsize = BATCH_SIZE
    if output_fname is not None:
        log_folder = os.path.dirname(output_fname)
    else:
        log_folder = os.path.join('results', 'mnist', 'offline', 'smd')
        log_folder = os.path.join(log_folder, _get_run_number(log_folder))
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    with tf.Graph().as_default(), tf.Session() as sess:
        dataset = get_dataset('mnist')
        exp_logger = _get_exp_logger(sess, log_folder)
        if effective_lr:
            init_lr_ = init_lr / float(1.0 - momentum)
        else:
            init_lr_ = init_lr

        if negative_momentum:
            init_mom_ = 1.0 - momentum
        else:
            init_mom_ = momentum

        config = get_mnist_mlp_config(
            init_lr_,
            init_mom_,
            decay=init_decay,
            effective_lr=effective_lr,
            negative_momentum=negative_momentum)
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
        y = tf.placeholder(tf.int64, [None], name="y")
        with tf.name_scope('Train'):
            with tf.variable_scope('Model'):
                model = get_mnist_mlp_model(
                    config, x, y, optimizer='momentum_inv_decay', training=True)
        all_vars = tf.global_variables()
        var_to_restore = list(filter(lambda x: 'momentum' not in x.name.lower(), all_vars))
        var_to_restore = list(filter(lambda x: 'global_step' not in x.name.lower(), var_to_restore))
        var_to_restore = list(filter(lambda x: 'lr' not in x.name.lower(), var_to_restore))
        var_to_restore = list(filter(lambda x: 'mom' not in x.name.lower(), var_to_restore))
        var_to_restore = list(filter(lambda x: 'decay' not in x.name.lower(), var_to_restore))
        saver = tf.train.Saver(var_to_restore)
        rnd = np.random.RandomState(seed)

        hp_dict = {'lr': init_lr, 'decay': init_decay}
        hp_names = hp_dict.keys()
        hyperparams = dict(
            [(hp_name, model.optimizer.hyperparams[hp_name]) for hp_name in hp_names])
        grads = model.optimizer.grads
        accumulators = model.optimizer.accumulators
        new_accumulators = model.optimizer.new_accumulators
        loss = model.cost

        # Build look ahead graph.
        look_ahead_ops, hp_grad_ops, zero_out_ops = look_ahead_grads(
            hyperparams, grads, accumulators, new_accumulators, loss)

        # Meta optimizer, use Adam on the log space.
        # meta_opt = LogOptimizer(tf.train.AdamOptimizer(meta_lr))
        meta_opt = LogOptimizer(tf.train.MomentumOptimizer(meta_lr, 0.9))
        hp = [model.optimizer.hyperparams[hp_name] for hp_name in hp_names]
        hp_grads_dict = {
            'lr': tf.placeholder(tf.float32, [], name='lr_grad'),
            'decay': tf.placeholder(tf.float32, [], name='decay_grad')
        }
        hp_grads_plh = [hp_grads_dict[hp_name] for hp_name in hp_names]
        hp_grads_and_vars = list(zip(hp_grads_plh, hp))
        cgrad = {'lr': (-1e1, 1e1), 'decay': (-1e1, 1e1)}
        cval = {'lr': (1e-4, 1e1), 'decay': (1e-4, 1e3)}
        cgrad_ = [cgrad[hp_name] for hp_name in hp_names]
        cval_ = [cval[hp_name] for hp_name in hp_names]
        meta_train_op = meta_opt.apply_gradients(
            hp_grads_and_vars, clip_gradients=cgrad_, clip_values=cval_)

        if output_fname is not None:
            msg = '{} exists, please remove previous experiment data.'.format(output_fname)
            assert not os.path.exists(output_fname), msg
            log.info('Writing to {}'.format(output_fname))
            with open(output_fname, 'w') as f:
                f.write('Step,LR,Mom,Decay,Loss\n')

        # Initialize all variables.
        sess.run(tf.global_variables_initializer())
        var_list = tf.global_variables()
        if pretrain_ckpt is not None:
            saver.restore(sess, pretrain_ckpt)
        ckpt = build_checkpoint(var_list)
        write_op = write_checkpoint(ckpt, var_list)
        read_op = read_checkpoint(ckpt, var_list)
        sess.run(write_op)

        # Progress bar.
        it = tqdm(
            six.moves.xrange(num_meta_steps),
            ncols=0,
            desc='look_{}_ilr_{:.0e}_decay_{:.0e}'.format(num_steps, init_lr, init_decay))

        for run in it:
            # Stochastic data list makes the SMD converge faster.
            data_list = [dataset.next_batch(bsize) for step in six.moves.xrange(num_steps)]
            eval_data_list = [
                dataset.next_batch(bsize) for step in six.moves.xrange(NUM_TRAIN // bsize)
            ]
            # Run meta descent step.
            cost, hp_dict = meta_step(sess, model, data_list, look_ahead_ops, hp_grad_ops,
                                      hp_grads_plh, meta_train_op, eval_data_list)

            # Early stop if hits NaN.
            if np.isnan(cost):
                break

            # Restore parameters.
            sess.run(read_op)
            for hpname, hpval in hp_dict.items():
                model.optimizer.assign_hyperparam(sess, hpname, hpval)

            # Read out hyperparameters in normal parameterization.
            if negative_momentum:
                mom = 1 - hp_dict['mom']
            else:
                mom = hp_dict['mom']
            if effective_lr:
                lr = hp_dict['lr'] * (1 - mom)
            else:
                lr = hp_dict['lr']

            # Write to logs.
            if output_fname is not None:
                with open(output_fname, 'a') as f:
                    f.write('{:d},{:f},{:f},{:f},{:f}\n'.format(run, lr, hp_dict['mom'],
                                                                hp_dict['decay'], cost))
            # Log to TensorBoard.
            exp_logger.log(run, 'lr', lr)
            exp_logger.log(run, 'decay', hp_dict['decay'])
            exp_logger.log(run, 'log loss', np.log10(cost))
            exp_logger.flush()

            # Update progress bar.
            it.set_postfix(
                lr='{:.3e}'.format(lr),
                decay='{:.3e}'.format(hp_dict['decay']),
                loss='{:.3e}'.format(cost))

        exp_logger.close()


def plot_meta_curve(folder, ax, color='r'):
    """Plots an SMD optimization curve on the surface.

    Args:
        folder: Folder where the SMD optimization log is stored.
    """
    data = pd.read_csv(os.path.join(folder, 'log.csv'))
    lr = data['LR']
    decay_exp = data['Decay']
    loss = data['Loss']
    final_step = np.argmin(loss)
    x = decay_exp[:final_step]
    y = lr[:final_step]
    print('Final step', final_step)
    ax.plot(x, y, color)
    # Show arrows.
    ss = 50
    x1 = np.ma.masked_array(x[::ss][:-1], np.diff(x[::ss]) >= 0)
    x2 = np.ma.masked_array(x[::ss][:-1], np.diff(x[::ss]) <= 0)
    ax.plot(x1, y[::ss][:-1], color + '>')
    ax.plot(x2, y[::ss][:-1], color + '<')
    ax.set_ylim(np.exp(np.log(10) * FLAGS.low_log_lr), np.exp(np.log(10) * FLAGS.high_log_lr))
    # Find the optimal learning rate and decay.
    return y[-20:].mean(), x[-20:].mean()


def main():
    assert FLAGS.run is not None, 'Need to specify --run flag.'
    # --------------------------------------------------------------------
    # Pretrain for a certain number of steps.
    pretrain(FLAGS.num_pretrain_steps)
    lr_limit = (FLAGS.low_log_lr, FLAGS.high_log_lr)
    decay_limit = (FLAGS.low_log_decay, FLAGS.high_log_decay)
    num_samples = FLAGS.num_samples

    # --------------------------------------------------------------------
    # Configurations.
    # Number of look ahead steps.
    step_list = [int(ss) for ss in FLAGS.num_steps.split(',')]
    surface_folder = os.path.join('results', 'mnist', 'offline', 'surface')
    fname_list = [
        os.path.join(surface_folder, 'offline_pt{}_look_{}.csv'.format(
            FLAGS.num_pretrain_steps, step)) for step in step_list
    ]

    # --------------------------------------------------------------------
    # Random search hyperparameters.
    if FLAGS.run == 'surface':
        for num_steps, fname in zip(step_list, fname_list):
            log.info('Running random search for look ahead step = {}'.format(num_steps))
            run_random_search(
                num_steps,
                lr_limit,
                decay_limit,
                num_samples,
                PRETRAIN_FILE,
                fname,
                seed=FLAGS.seed)

    # --------------------------------------------------------------------
    # Plot hyperparameter surface.
    if FLAGS.run in ['best', 'surface', 'plot']:
        axes, best_alpha, best_decay = plot_surface(fname_list, step_list)
        plt.savefig(
            os.path.join(surface_folder, 'offline_pt{}_surface.pdf'.format(
                FLAGS.num_pretrain_steps)))
        print('Best alpha', best_alpha)
        print('Best decay', best_decay)

    # --------------------------------------------------------------------
    # Run the best hyperparameter settings.
    if FLAGS.run == 'best':
        lr_list = best_alpha
        lr_list = [x * 0.1 for x in lr_list]
        decay_list = best_decay
        best_folder = os.path.join('results', 'mnist', 'offline', 'best')
        if not os.path.exists(best_folder):
            os.makedirs(best_folder)
        output_list = [
            os.path.join(best_folder, 'offline_pt{}_look_{}_best.npy'.format(
                FLAGS.num_pretrain_steps, step)) for step in step_list
        ]

        if not all([os.path.exists(output) for output in output_list]):
            # run_final_setup(lr_list[1:], decay_list[1:], output_list[1:])
            run_final_setup(lr_list, decay_list, output_list)
        else:
            print('Results found. Skip running setups.')
        # --------------------------------------------------------------------
        # Plot final setup.
        plot_final_setup(output_list, lr_list, decay_list, step_list,
                         os.path.join('results', 'mnist', 'offline', 'best',
                                      'offline_pt{}_best.pdf'.format(FLAGS.num_pretrain_steps)))

    # --------------------------------------------------------------------
    # Offline SMD
    if FLAGS.run == 'smd':
        axes, best_alpha, best_decay = plot_surface(fname_list, step_list)

        init_lr_list = [float(ss) for ss in FLAGS.init_lr.split(',')]
        init_decay_list = [float(ss) for ss in FLAGS.init_decay.split(',')]
        step_idx = 0  # Which look ahead step we are at.
        step_list = list(filter(lambda x: x <= 5000, step_list))  # Let's ignore 20k steps for now.
        smd_folder = os.path.join('results', 'mnist', 'offline', 'smd')
        for jj, step in enumerate(step_list):
            for init_lr, init_decay in itertools.product(init_lr_list, init_decay_list):
                log.info(
                    'Run SMD with look ahead {} steps, init lr={:.3f}, init decay={:.3f}'.format(
                        step, init_lr, init_decay))
                output_folder = os.path.join(smd_folder, 'look_{:d}_lr_{:.0e}_decay_{:.0e}'.format(
                    step, init_lr, init_decay))

                if os.path.exists(output_folder):
                    log.info('Found folder {} exists, skip running.'.format(output_folder))
                else:
                    output_fname = os.path.join(output_folder, 'log.csv')
                    run_offline_smd(
                        step,
                        init_lr,
                        init_decay,
                        FLAGS.meta_lr,
                        FLAGS.num_meta_steps,
                        momentum=MOMENTUM,
                        effective_lr=True,
                        negative_momentum=False,
                        pretrain_ckpt=PRETRAIN_FILE,
                        output_fname=output_fname)

                # Plot meta-optimization curve on the loss surface.
                best_alpha, best_decay = plot_meta_curve(output_folder, axes[jj])
                print('Optimal selection of offline SMD', step, best_alpha, best_decay)
                step_idx += 1
        plot_fname = os.path.join(surface_folder, 'offline_pt{}_surface_traj.pdf'.format(
            FLAGS.num_pretrain_steps))
        plt.savefig(plot_fname)
        log.info('SMD plot saved: {}'.format(plot_fname))


if __name__ == '__main__':
    main()
