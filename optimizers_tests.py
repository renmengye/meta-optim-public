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
Unit tests for optimizers.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import six
import tensorflow as tf

from optimizers import AdamOptimizer
from optimizers import GradientDescentOptimizer
from optimizers import LogOptimizer
from optimizers import MomentumInvDecayOptimizer
from optimizers import MomentumOptimizer


class NumpyOptimizer(object):
    def minimize(self, loss, var_list=None, gate_gradients=1):
        self._loss = loss
        self._var_list = var_list
        if var_list is None:
            var_list = tf.trainable_variables()
        self._grads = tf.gradients(loss, var_list, gate_gradients=gate_gradients)

    def run_minimize(self, sess, feed_dict=None):
        def _loss_grad_func(val_list):
            sess.run([tf.assign(vv, val) for vv, val in zip(self._var_list, val_list)])
            r = sess.run([self._loss] + self._grads, feed_dict=feed_dict)
            return r[0], r[1:]

        new_val_list = self._minimize(sess.run(self._var_list), _loss_grad_func)
        sess.run([tf.assign(vv, val) for vv, val in zip(self._var_list, new_val_list)])

    def _minimize(self):
        raise NotImplemented


class AdamNumpyOptimizer(NumpyOptimizer):
    t = 0

    def __init__(self, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def minimize(self, loss, var_list=None, gate_gradients=1):
        if var_list is None:
            var_list = tf.trainable_variables()
        self.m = [np.zeros(x.shape) for x in var_list]
        self.v = [np.zeros(x.shape) for x in var_list]
        super(AdamNumpyOptimizer, self).minimize(
            loss, var_list=var_list, gate_gradients=gate_gradients)

    def _minimize(self, initial_val, loss_grad_func):
        _, grads = loss_grad_func(initial_val)
        next_val = []
        self.t += 1
        alpha_t = self.alpha * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for jj, (grad, param) in enumerate(zip(grads, initial_val)):
            self.m[jj] = self.beta1 * self.m[jj] + (1 - self.beta1) * grad
            self.v[jj] = self.beta2 * self.v[jj] + (1 - self.beta2) * grad * grad
            param_t = param - alpha_t * self.m[jj] / (np.sqrt(self.v[jj]) + self.epsilon)
            next_val.append(param_t)
        return next_val


class OptimizersTests(tf.test.TestCase):
    _dtype = tf.float64
    _npdtype = np.float64
    _num_steps = 10
    _rtol = 1e-5
    _atol = 1e-6

    def _test_numpy_optimizer_equal(self, sess, opt_a, opt_b, rnd=None):
        if rnd is None:
            rnd = np.random.RandomState(0)
        w_init = rnd.uniform(-0.1, 0.1, [100, 1]).astype(self._npdtype)
        w = tf.get_variable(
            'w', initializer=tf.constant(w_init, dtype=self._dtype), dtype=self._dtype)
        x = tf.constant(rnd.uniform(-1, 1, [50, 100]).astype(self._npdtype), dtype=self._dtype)
        y_noise = tf.constant(rnd.uniform(-0.1, 0.1, [50]).astype(self._npdtype), dtype=self._dtype)
        w_true = tf.constant(
            rnd.uniform(-0.1, 0.1, [100, 1]).astype(self._npdtype), dtype=self._dtype)
        y = tf.matmul(x, w_true) + y_noise
        y_pred = tf.matmul(x, w)
        cost = tf.reduce_mean(tf.square(y - y_pred))
        grad_w = tf.gradients(cost, [w], gate_gradients=2)[0]
        train_op_a = opt_a.minimize(cost, gate_gradients=2)
        opt_b.minimize(cost, gate_gradients=2)

        # Optimizer A.
        sess.run(tf.global_variables_initializer())
        w_list_a = [sess.run(w)]
        delta_a = []
        grad_w_a = []
        for ii in six.moves.xrange(self._num_steps):
            grad_w_a.append(sess.run(grad_w))
            sess.run(train_op_a)
            w_new = sess.run(w)
            delta_a.append(w_new - w_list_a[-1])
            w_list_a.append(w_new)

        # Optimizer B.
        sess.run(tf.variables_initializer([w]))
        w_list_b = [sess.run(w)]
        delta_b = []
        grad_w_b = []
        for ii in six.moves.xrange(self._num_steps):
            grad_w_b.append(sess.run(grad_w))
            opt_b.run_minimize(sess)
            w_new = sess.run(w)
            delta_b.append(w_new - w_list_b[-1])
            w_list_b.append(w_new)

        for wa, wb in list(zip(grad_w_a, grad_w_b)):
            np.testing.assert_allclose(wa, wb, rtol=self._rtol, atol=self._atol)

        for wa, wb in list(zip(w_list_a, w_list_b)):
            np.testing.assert_allclose(wa, wb, rtol=self._rtol, atol=self._atol)

    def _test_optimizer_equal(self, sess, opt_a, opt_b, rnd=None):
        if rnd is None:
            rnd = np.random.RandomState(0)
        w_init = rnd.uniform(-0.1, 0.1, [100, 1]).astype(self._npdtype)
        w = tf.get_variable(
            'w', initializer=tf.constant(w_init, dtype=self._dtype), dtype=self._dtype)
        x = tf.constant(rnd.uniform(-1, 1, [50, 100]).astype(self._npdtype), dtype=self._dtype)
        y_noise = tf.constant(rnd.uniform(-0.1, 0.1, [50]).astype(self._npdtype), dtype=self._dtype)
        w_true = tf.constant(
            rnd.uniform(-0.1, 0.1, [100, 1]).astype(self._npdtype), dtype=self._dtype)
        y = tf.matmul(x, w_true) + y_noise
        y_pred = tf.matmul(x, w)
        cost = tf.reduce_mean(tf.square(y - y_pred))
        grad_w = tf.gradients(cost, [w], gate_gradients=2)[0]
        train_op_a = opt_a.minimize(cost, gate_gradients=2)
        train_op_b = opt_b.minimize(cost, gate_gradients=2)

        # Optimizer A.
        sess.run(tf.global_variables_initializer())
        w_list_a = [sess.run(w)]
        delta_a = []
        grad_w_a = []
        for ii in six.moves.xrange(self._num_steps):
            grad_w_a.append(sess.run(grad_w))
            sess.run(train_op_a)
            w_new = sess.run(w)
            delta_a.append(w_new - w_list_a[-1])
            w_list_a.append(w_new)

        # Optimizer B.
        sess.run(tf.variables_initializer([w]))
        w_list_b = [sess.run(w)]
        delta_b = []
        grad_w_b = []
        for ii in six.moves.xrange(self._num_steps):
            grad_w_b.append(sess.run(grad_w))
            sess.run(train_op_b)
            w_new = sess.run(w)
            delta_b.append(w_new - w_list_b[-1])
            w_list_b.append(w_new)

        for wa, wb in list(zip(grad_w_a, grad_w_b)):
            np.testing.assert_allclose(wa, wb, rtol=self._rtol, atol=self._atol)

        for wa, wb in list(zip(w_list_a, w_list_b)):
            np.testing.assert_allclose(wa, wb, rtol=self._rtol, atol=self._atol)

    def test_gradient_descent_optimizer(self):
        """Test gradient descent optimizer."""
        lr = 1e-2
        with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
            self._test_optimizer_equal(sess, tf.train.GradientDescentOptimizer(lr),
                                       GradientDescentOptimizer(lr, dtype=self._dtype))

    def test_momentum_optimizer(self):
        """Test momentum optimizer."""
        lr = 1e-2
        mom = 9e-1
        with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
            self._test_optimizer_equal(sess, tf.train.MomentumOptimizer(lr, mom),
                                       MomentumOptimizer(lr, mom, dtype=self._dtype))

    def test_adam_optimizer(self):
        """Test momentum optimizer."""
        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
            self._test_optimizer_equal(sess,
                                       tf.train.AdamOptimizer(
                                           lr, beta1=beta1, beta2=beta2, epsilon=epsilon),
                                       AdamOptimizer(
                                           lr,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon,
                                           dtype=self._dtype))

    def test_adam_optimizer_numpy(self):
        """Test momentum optimizer."""
        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
            self._test_numpy_optimizer_equal(sess,
                                             tf.train.AdamOptimizer(
                                                 lr, beta1=beta1, beta2=beta2, epsilon=epsilon),
                                             AdamNumpyOptimizer(
                                                 lr, beta1=beta1, beta2=beta2, epsilon=epsilon))

        with tf.Graph().as_default(), tf.Session() as sess, tf.device("/cpu:0"):
            self._test_optimizer_equal(sess,
                                       tf.train.AdamOptimizer(
                                           lr, beta1=beta1, beta2=beta2, epsilon=epsilon),
                                       AdamOptimizer(
                                           lr,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon,
                                           dtype=self._dtype))


class LogOptimizerTests(tf.test.TestCase):
    def test_log_optimizer(self):
        """Tests optimizers on log space."""
        with tf.Graph().as_default(), tf.device("/cpu:0"):
            x = tf.Variable(1.0)
            f = tf.square(x)
            opt = LogOptimizer(tf.train.GradientDescentOptimizer(1.0))
            grads = tf.gradients(f, x)
            grads_and_vars = zip(grads, [x])
            op = opt.apply_gradients(grads_and_vars, clip_values=None, clip_gradients=None)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(op)
                np.testing.assert_allclose(np.exp(-2.0), x.eval())

    def test_log_clip_value_optimizer(self):
        """Tests optimizers on log space, with value clipping."""
        with tf.Graph().as_default(), tf.device("/cpu:0"):
            x = tf.Variable(1.0)
            f = tf.square(x)
            opt = LogOptimizer(tf.train.GradientDescentOptimizer(1.0))
            grads = tf.gradients(f, x)
            grads_and_vars = zip(grads, [x])
            op = opt.apply_gradients(grads_and_vars, clip_values=[(-0.1, 0.1)], clip_gradients=None)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(op)
                np.testing.assert_allclose(0.1, x.eval())

    def test_log_clip_grads_optimizer(self):
        """Tests optimizers on log space, with gradient clipping."""
        with tf.Graph().as_default(), tf.device("/cpu:0"):
            x = tf.Variable(1.0)
            f = tf.square(x)
            opt = LogOptimizer(tf.train.GradientDescentOptimizer(1.0))
            grads = tf.gradients(f, x)
            grads_and_vars = zip(grads, [x])
            op = opt.apply_gradients(grads_and_vars, clip_values=None, clip_gradients=[(-0.1, 0.1)])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(op)
                np.testing.assert_allclose(np.exp(-0.1), x.eval())


if __name__ == '__main__':
    tf.test.main()
