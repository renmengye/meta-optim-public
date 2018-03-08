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
"""
Look ahead forward mode gradients.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow_forward_ad import forward_gradients


def zero_out(var_list, use_locking=False):
    """Zeros out a list of variables.
    Note: This is hyperparameter-agnostic, can be used on SGD and SGD+Momentum.

    Args:
      var_list: List of variables.

    Returns:
      zero_out_list: List of zero out operations.
    """
    return list(map(lambda x: tf.assign(x, tf.zeros_like(x), use_locking=use_locking), var_list))


def look_ahead_grads(hyperparams,
                     grads,
                     accumulators,
                     new_accumulators,
                     loss,
                     parameter_key='w',
                     dtype=tf.float32):
    """Returns the gradients for the hyperparameters.

    Args:
        hyperparams: A dictionary of hyperparameters. E.g. {'lr': lr, 'mom': mom}.
        grads: A list of gradients to regular parameters.
        accumulators: A dictionary of lists of accumulators such as weights and velocities, same
            order as `grads`. E.g. {'w': var_list, 'v': velocity}.
        new_accumulators: A dictionary of lists of accumators for the next step, same order as
            `grads`. E.g. {'w': var_list_new, 'v': velocity_new}.
        loss: Loss function.
        parameter_key: Key in accumulator that represents the weights parameters, default `w`.
        dtype: Data dtype, default tf.float32.

    Returns:
        look_ahead_ops: Accumulates the gradients from accumulators to hyperparameters.
        grad_ops: Gets the gradients from loss to current step.
        zero_out_ops: Clear out the gradients from accumulators to hyperparameters.
    """
    # Initialize gradient variables.
    get_name = lambda x: x.name.split('/')[-1].split(':')[0]
    grad_acc_vars = {}
    grad_loss_var = {}
    hp_keys = hyperparams.keys()
    acc_keys = accumulators.keys()
    for _hp_key in hyperparams:
        hp = hyperparams[_hp_key]
        grad_acc_vars[_hp_key] = {}
        for _acc_key in acc_keys:
            acc = accumulators[_acc_key]
            with tf.variable_scope("grad_{}_{}".format(_hp_key, _acc_key)):
                # dw_da
                grad_acc_vars[_hp_key][_acc_key] = list(map(
                    lambda x: tf.get_variable(get_name(x), x.get_shape(),
                      dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype), trainable=False),
                    acc))
        # Calculate dE_dalpha_t, with dw_dalpha_t and dE_dw_t
        grad_loss_var[_hp_key] = tf.get_variable(
            "grad_loss_{}".format(_hp_key), [],
            dtype=dtype,
            initializer=tf.constant_initializer(0.0, dtype=dtype),
            trainable=False)

    all_accumulators = []
    for _acc_key in acc_keys:
        acc = accumulators[_acc_key]
        all_accumulators += acc

    new_grad_acc_vars = {}
    zero_out_ops = []
    for _hp_key in hp_keys:
        hp = hyperparams[_hp_key]
        all_grad_acc_vars = []
        new_grad_acc_vars[_hp_key] = {}
        for _acc_key in acc_keys:
            all_grad_acc_vars += grad_acc_vars[_hp_key][_acc_key]

        for _acc_key in acc_keys:
            new_grad_acc_vars[_hp_key][_acc_key] = forward_gradients(
                new_accumulators[_acc_key], [hp] + all_accumulators,
                grad_xs=[tf.constant(1.0, dtype=dtype)] + all_grad_acc_vars,
                gate_gradients=True)

        zero_out_ops.extend(zero_out(all_grad_acc_vars))

    look_ahead_ops = [[[
        tf.assign(ga, ga_new) for ga, ga_new in zip(grad_acc_vars[_hp_key][_acc_key],
                                                    new_grad_acc_vars[_hp_key][_acc_key])
    ] for _hp_key in hp_keys] for _acc_key in acc_keys]
    look_ahead_ops = list(np.array(look_ahead_ops).flatten())

    grad_loss = {}
    for _hp_key in hp_keys:
        grad_loss[_hp_key] = tf.reduce_sum(
            [tf.reduce_sum(g * a) for g, a in zip(grads, grad_acc_vars[_hp_key][parameter_key])])
    grad_ops = [tf.assign(grad_loss_var[_hp_key], grad_loss[_hp_key]) for _hp_key in hp_keys]
    return look_ahead_ops, grad_ops, zero_out_ops
