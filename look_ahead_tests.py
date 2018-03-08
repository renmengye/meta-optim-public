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
Unit tests for look ahead forward mode gradients.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from checkpoint import build_checkpoint, read_checkpoint, write_checkpoint
from models import mlp, get_mnist_mlp_model, get_mnist_mlp_config
from look_ahead import look_ahead_grads


class LookAheadTests(tf.test.TestCase):
    _dtype = tf.float64
    _npdtype = np.float64
    _lr = 1e-1
    _momentum = 9e-1
    _decay = 3.0
    _eps = 1e-7
    _batch_size = 100
    _input_size = [28, 28, 1]
    _rtol = 1e-5
    _atol = 1e-6
    _lookahead_steps = 5

    def get_batch_random(self, batch_size=100, rnd=None):
        """Gets a random batch."""
        if rnd is None:
            rnd = np.random.RandomState(0)
        return (rnd.uniform(-1, 1, [batch_size] + self._input_size).astype(self._npdtype),
                np.floor(rnd.uniform(0, 10, [batch_size])).astype(np.int64))

    def get_fdict(self, model, batch):
        """Gets a feed dict."""
        x = batch[0].astype(self._npdtype)
        y = batch[1].astype(np.int64)
        fdict = {model.x: x, model.y: y}
        return fdict

    def run_loss(self, sess, model, batch_size, look_ahead_ops, grad_ops, neval=1):
        """Gets the loss for the t+k step.

        Args:
            sess: TensorFlow session object.
            model: Model object.
            batch_size: Int. Mini-batch size.
            look_ahead_ops: Ops for accumulating look ahead gradients.
            grad_ops: Ops for fetching the gradients to the hyperparameters.
            neval: Number of evaluation batches, default 1.

        Returns:
            cost: Float. Cost value.
            grads: List of floats. Gradients for each of the hyperparameters.
        """
        rnd = np.random.RandomState(1234)
        batches = [
            self.get_batch_random(batch_size, rnd) for ii in range(self._lookahead_steps + neval)
        ]
        for ii in range(self._lookahead_steps):
            fdict = self.get_fdict(model, batches[ii])
            sess.run(look_ahead_ops, feed_dict=fdict)
            sess.run(model.train_op, feed_dict=fdict)
        cost = 0.0
        grads = [0.0] * len(grad_ops)
        for ii in range(neval):
            fdict = self.get_fdict(model, batches[self._lookahead_steps + ii])
            results = sess.run([model.cost] + grad_ops, feed_dict=fdict)
            cost += results[0] / float(neval)
            for jj, rr in enumerate(results[1:]):
                grads[jj] += rr / float(neval)
        return cost, grads

    def get_model(self, optimizer='momentum', effective_lr=False, negative_momentum=False):
        """Builds a model.

        Args:
            optimizer: String. Name of the optimizer.
            effective_lr: Bool. Whether to optimize using the effective learning rate.
            negative momentum: Bool. Whether to optimize using the negative momentum.
        """
        config = get_mnist_mlp_config(
            self._lr,
            self._momentum,
            decay=self._decay,
            effective_lr=effective_lr,
            negative_momentum=negative_momentum)
        x = tf.placeholder(self._dtype, [self._batch_size] + self._input_size)
        y = tf.placeholder(tf.int64, [self._batch_size])
        model = get_mnist_mlp_model(config, x, y, optimizer=optimizer, dtype=self._dtype)
        return model

    def _test_ad(self, sess, model, hp_dict, neval):
        """Test automatic differentiation.

        Args:
            model: Mode object.
            hp_dict: A dictionary of hyperparameter name and initial values.
        """
        hyperparams = dict(
            [(hp_name, model.optimizer.hyperparams[hp_name]) for hp_name in hp_dict.keys()])
        grads = model.optimizer.grads
        accumulators = model.optimizer.accumulators
        new_accumulators = model.optimizer.new_accumulators
        loss = model.cost

        # Build look ahead graph.
        look_ahead_ops, grad_ops, zero_out_ops = look_ahead_grads(
            hyperparams, grads, accumulators, new_accumulators, loss, dtype=self._dtype)

        # Gets variables to be checkpointed.
        ckpt_var_list = []
        for var_list in model.optimizer.accumulators.values():
            ckpt_var_list.extend(var_list)
        ckpt_var_list.append(model.global_step)

        # Build checkpoint ops.
        ckpt = build_checkpoint(ckpt_var_list)
        read_checkpoint_op = read_checkpoint(ckpt, ckpt_var_list)
        write_checkpoint_op = write_checkpoint(ckpt, ckpt_var_list)

        # Initialize weights parameters.
        sess.run(tf.global_variables_initializer())

        # Checkpoint weights and momentum parameters.
        sess.run(write_checkpoint_op)

        # Initialize hyperparameters.
        for hp_name, init_hp in hp_dict.items():
            model.optimizer.assign_hyperparam(sess, hp_name, init_hp)
        _, grad_hp = self.run_loss(sess, model, self._batch_size, look_ahead_ops, grad_ops)

        # Check the gradient of loss wrt. each hp.
        for ii, (hp_name, init_hp) in enumerate(hp_dict.items()):
            sess.run(read_checkpoint_op)
            sess.run(zero_out_ops)
            model.optimizer.assign_hyperparam(sess, hp_name, init_hp - self._eps)
            l1, _ = self.run_loss(sess, model, self._batch_size, look_ahead_ops, grad_ops)
            sess.run(read_checkpoint_op)
            sess.run(zero_out_ops)
            model.optimizer.assign_hyperparam(sess, hp_name, init_hp + self._eps)
            l2, _ = self.run_loss(sess, model, self._batch_size, look_ahead_ops, grad_ops)
            grad_hp_fd = (l2 - l1) / (2 * self._eps)
            model.optimizer.assign_hyperparam(sess, hp_name, init_hp)
            np.testing.assert_allclose(grad_hp[ii], grad_hp_fd, rtol=self._rtol, atol=self._atol)

    def test_grad_gradient_descent_optimizer(self):
        """Tests the gradients of a gradient descent optimizer."""
        for neval in [1, 2, 5]:
            with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
                opt = 'gradient_descent'
                m = self.get_model(optimizer=opt)
                self._test_ad(sess, m, {'lr': self._lr}, neval)

    def test_grad_momentum_optimizer(self):
        """Tests the gradients of a momentum optimizer."""
        for neval in [1, 2, 5]:
            for eff_lr in [True, False]:
                for neg_mom in [True, False]:
                    if eff_lr:
                        lr0 = self._lr / (1 - self._momentum)
                    else:
                        lr0 = self._lr
                    if neg_mom:
                        mom0 = 1 - self._momentum
                    else:
                        mom0 = self._momentum
                    with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
                        opt = 'momentum'
                        m = self.get_model(
                            optimizer=opt, effective_lr=eff_lr, negative_momentum=neg_mom)
                        self._test_ad(sess, m, {'lr': lr0, 'mom': mom0}, neval)

    def test_grad_momentum_inv_decay_optimizer(self):
        """Tests the gradients of a momentum inverse decay optimizer."""
        for neval in [1, 2, 5]:
            for eff_lr in [True, False]:
                for neg_mom in [True, False]:
                    if eff_lr:
                        lr0 = self._lr / (1 - self._momentum)
                    else:
                        lr0 = self._lr
                    if neg_mom:
                        mom0 = 1 - self._momentum
                    else:
                        mom0 = self._momentum
                    with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
                        opt = 'momentum_inv_decay'
                        m = self.get_model(
                            optimizer=opt, effective_lr=eff_lr, negative_momentum=neg_mom)
                        self._test_ad(sess, m, {'lr': lr0,
                                                'mom': mom0,
                                                'decay': self._decay}, neval)

    def test_grad_adam_optimizer(self):
        """Tests the gradients of an Adam optimizer."""
        for neval in [1, 2, 5]:
            with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
                opt = 'adam'
                m = self.get_model(optimizer=opt)
                self._test_ad(sess, m, {'lr': self._lr}, neval)


if __name__ == '__main__':
    tf.test.main()
