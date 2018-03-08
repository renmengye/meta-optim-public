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
"""Training utilities.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import six
import tensorflow as tf

from collections import namedtuple
from tqdm import tqdm

from get_dataset import get_dataset
from logger import get as get_logger
from models import get_mnist_mlp_config, get_mnist_mlp_model

log = get_logger()

# Training curves.
Results = namedtuple('Results',
                     ['step', 'train_xent', 'train_acc', 'test_xent', 'test_acc', 'lr', 'decay'])


def save_results(fname, results):
    """Saves training results."""
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    np.save(fname, np.array(results._asdict(), dtype=object))


def train_steps(sess, m, data_list, init_lr=0.1, decay_const=0.0, time_const=5000.0):
    """Train an MLP for MNIST for certain amount of steps.

    Args:
        sess: TensorFlow session object.
        m: Model object.
        data_list: List of tuples of x and y's.
        init_lr: Float. Initial learning rate.
        decay: Float. Decay constant.

    Returns:
        cost: Final cost by the end of the training.
    """
    for ii, (xd, yd) in enumerate(data_list):
        if decay_const > 0.0:
            lr_ = init_lr / ((1.0 + ii / time_const)**decay_const)
        m.optimizer.assign_hyperparam(sess, 'lr', lr_)
        if lr_ > 1e-6:
            cost_, _ = sess.run([m.cost, m.train_op], feed_dict={m.x: xd, m.y: yd})

    final_cost = 0.0
    for ii, (xd, yd) in enumerate(data_list[:600]):
        final_cost += sess.run(m.cost, feed_dict={m.x: xd, m.y: yd}) / 600.0
    return cost_, final_cost


def train_mnist_mlp_with_test(init_lr=0.1,
                              momentum=0.9,
                              num_steps=50000,
                              middle_decay=False,
                              inverse_decay=False,
                              decay_const=0.0,
                              time_const=5000.0,
                              steps_per_eval=100,
                              batch_size=100,
                              pretrain_ckpt=None,
                              save_ckpt=None,
                              print_step=False,
                              data_list=None,
                              data_list_eval=None,
                              data_list_test=None):
    """Train an MLP for MNIST.

    Args:
        init_lr:
        momentum:
        num_steps:
        middle_decay:
        pretrain_ckpt:

    Returns:
        results: Results tuple object.
    """
    if data_list is None:
        dataset = get_dataset('mnist')
    if data_list_eval is None:
        dataset_train = get_dataset('mnist')
    if data_list_test is None:
        dataset_test = get_dataset('mnist', test=True)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
    y = tf.placeholder(tf.int64, [None], name="y")
    config = get_mnist_mlp_config(init_lr, momentum)
    with tf.name_scope('Train'):
        with tf.variable_scope('Model'):
            m = get_mnist_mlp_model(config, x, y, training=True)
    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            mtest = get_mnist_mlp_model(config, x, y, training=False)

    final_lr = 1e-4
    midpoint = num_steps // 2

    if True:
        num_train = 60000
        num_test = 10000
    lr_ = init_lr
    bsize = batch_size
    steps_per_epoch = num_train // bsize
    steps_test_per_epoch = num_test // bsize
    tau = (num_steps - midpoint) / np.log(init_lr / final_lr)

    train_xent_list = []
    train_cost_list = []
    train_acc_list = []
    test_xent_list = []
    test_cost_list = []
    test_acc_list = []
    lr_list = []
    step_list = []
    var_to_restore = list(filter(lambda x: 'momentum' not in x.name.lower(), tf.global_variables()))
    var_to_restore = list(filter(lambda x: 'global_step' not in x.name.lower(), var_to_restore))
    var_to_restore = list(filter(lambda x: 'lr' not in x.name.lower(), var_to_restore))
    var_to_restore = list(filter(lambda x: 'mom' not in x.name.lower(), var_to_restore))
    var_to_restore = list(filter(lambda x: 'decay' not in x.name.lower(), var_to_restore))
    var_to_init = list(filter(lambda x: x not in var_to_restore, tf.global_variables()))
    restorer = tf.train.Saver(var_to_restore)
    if inverse_decay:
        log.info(
            'Applying inverse decay with time constant = {:.3e} and decay constant = {:.3e}'.format(
                time_const, decay_const))
    if middle_decay:
        log.info('Applying decay at midpoint with final learning rate = {:.3e}'.format(final_lr))
    assert not (inverse_decay and
                middle_decay), 'Inverse decay and middle decay cannot be applied at the same time.'

    with tf.Session() as sess:
        if pretrain_ckpt is None:
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.variables_initializer(var_to_init))
            restorer.restore(sess, pretrain_ckpt)
        # Assign initial learning rate.
        m.optimizer.assign_hyperparam(sess, 'lr', lr_)
        train_iter = six.moves.xrange(num_steps)
        if not print_step:
            train_iter = tqdm(train_iter, ncols=0)

        for ii in train_iter:
            if data_list is None:
                xd, yd = dataset.next_batch(bsize)
            else:
                xd, yd = data_list[ii]
            if lr_ > 1e-6:
                cost_, _ = sess.run([m.cost, m.train_op], feed_dict={x: xd, y: yd})
            test_acc = 0.0
            test_xent = 0.0
            train_acc = 0.0
            train_xent = 0.0
            epoch = ii // steps_per_epoch

            if inverse_decay:
                lr_ = init_lr / ((1.0 + ii / time_const)**decay_const)

            if middle_decay and ii > midpoint:
                lr_ = np.exp(-(ii - midpoint) / tau) * init_lr

            m.optimizer.assign_hyperparam(sess, 'lr', lr_)

            # Evaluate every certain number of steps.
            if ii == 0 or (ii + 1) % steps_per_eval == 0:
                for jj in six.moves.xrange(steps_per_epoch):
                    if data_list_eval is None:
                        xd, yd = dataset_train.next_batch(bsize)
                    else:
                        xd, yd = data_list_eval[jj]
                    xent_, acc_ = sess.run([m.cost, m.acc], feed_dict={x: xd, y: yd})
                    train_xent += xent_ / float(steps_per_epoch)
                    train_acc += acc_ / float(steps_per_epoch)
                step_list.append(ii + 1)
                train_xent_list.append(train_xent)
                train_acc_list.append(train_acc)

                if data_list_eval is None:
                    dataset_train.reset()

                for jj in six.moves.xrange(steps_test_per_epoch):
                    if data_list_test is None:
                        xd, yd = dataset_test.next_batch(bsize)
                    else:
                        xd, yd = data_list_test[jj]
                    xent_, acc_ = sess.run([mtest.cost, mtest.acc], feed_dict={x: xd, y: yd})
                    test_xent += xent_ / float(steps_test_per_epoch)
                    test_acc += acc_ / float(steps_test_per_epoch)
                test_xent_list.append(test_xent)
                test_acc_list.append(test_acc)

                if data_list_test is None:
                    dataset_test.reset()

                lr_list.append(lr_)
                if print_step:
                    log.info(('Steps {:d} T Xent {:.3e} T Acc {:.3f} V Xent {:.3e} V Acc {:.3f} '
                              'LR {:.3e}').format(ii + 1, train_xent, train_acc * 100.0, test_xent,
                                                  test_acc * 100.0, lr_))
        if save_ckpt is not None:
            saver = tf.train.Saver()
            saver.save(sess, save_ckpt)

    return Results(
        step=np.array(step_list),
        train_xent=np.array(train_xent_list),
        train_acc=np.array(train_acc_list),
        test_xent=np.array(test_xent_list),
        test_acc=np.array(test_acc_list),
        lr=np.array(lr_list),
        decay=decay_const)


def meta_step(sess, model, data_list, look_ahead_ops, hp_grads_op, hp_grads_plh, meta_train_op,
              eval_data_list):
    """Run a meta step.

    Args:
        model: Model
        data_list: List of tuples of inputs and labels.
        look_ahead_ops: TensorFlow ops that accumulates hyperparameter gradients.
        hp_grads_op: TensorFlow ops to calculate the final hyperparameter gradients.
        hp_grads_plh: Placeholders for hyperparameter gradients.
        meta_train_op: TensorFlow ops that updates the hyperparameters.

    Returns:
        cost: Loss of the network at the end of the look ahead steps.
        hp: A dictionary maps from hyperparameter names to their current values.
    """
    assert len(data_list) > 1, 'We need to look ahead more than 1 step.'
    # Run till the second last item.
    for ii, (xd, yd) in enumerate(data_list[:-1]):
        fdict = {model.x: xd, model.y: yd}
        sess.run(look_ahead_ops, feed_dict=fdict)
        sess.run(model.train_op, feed_dict=fdict)
    cost = 0.0
    grads = [0.0] * len(hp_grads_plh)
    neval = len(eval_data_list)
    for ii, (xd, yd) in enumerate(eval_data_list):
        fdict = {model.x: xd, model.y: yd}
        results = sess.run([model.cost] + hp_grads_op, feed_dict=fdict)
        cost += results[0] / float(neval)
        for jj, rr in enumerate(results[1:]):
            grads[jj] += rr / float(neval)
    hp_fict = dict(zip(hp_grads_plh, grads))
    sess.run(meta_train_op, feed_dict=hp_fict)
    hp = sess.run(model.optimizer.hyperparams)
    return cost, hp
