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
Basic Optimizers.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def create_velocity_variables(var_list, name='momentum'):
    """Initialize momentum variables.

    Args:
        var_list: List of variables to be optimized.

    Returns:
        velocity: List of velocity variables.
    """

    def get_var(x):
        return tf.get_variable(
            x.name.split(':')[0],
            x.shape,
            initializer=tf.constant_initializer(0.0, dtype=x.dtype),
            dtype=x.dtype,
            trainable=False)

    with tf.variable_scope(name):
        velocity = list(map(get_var, var_list))
    return velocity


def gradient_descent(grads, var_list, lr):
    """Gradient descent update.

    Args:
        grads: List of gradients of the trainable variables.
        var_list: List of variables to be optimized.
        lr: Learning rate.

    Returns:
        var_list_new: List of new variables to be assigned.
    """
    return [var - lr * grad for grad, var in list(zip(grads, var_list))]


def gradient_descent_momentum(grads, velocity, var_list, lr, mom):
    """Gradient descent with momentum update.

    Args:
        grads: List of gradients of the trainable variables.
        velocity: List of velocity of the trainable variables.
        var_list: List of variables to be optimized.
        lr: Learning rate.
        mom: Momentum.

    Returns:
        var_list_new: List of new variables to be assigned.
        velocity_new: List of new velocity to be assigned.
    """
    velocity_new = [
        mom * v + grad
        for grad, v, var in list(zip(grads, velocity, var_list))
    ]
    var_list_new = [
        var - lr * v for v, var in list(zip(velocity_new, var_list))
    ]
    return var_list_new, velocity_new


def adam(grads, velocity_m, velocity_v, var_list, lr, beta1, beta2, epsilon):
    """ADAM update.

    Args:
        grads: List of gradients of the trainable variables.
        velocity_m: List of velocity of the trainable variables.
        velocity_v: List of velocity of the trainable variables.
        var_list: List of variables to be optimized.
        lr: Learning rate.
        beta1: First momentum.
        beta2: Second momentum.

    Returns:
        var_list_new: List of new variables to be assigned.
        velocity_m_new: List of new velocity_m to be assigned.
        velocity_v_new: List of new velocity_v to be assigned.
    """
    velocity_m_new = [
        beta1 * mm + (1 - beta1) * gg
        for gg, mm in list(zip(grads, velocity_m))
    ]
    velocity_v_new = [
        beta2 * vv + (1 - beta2) * gg * gg
        for gg, vv in list(zip(grads, velocity_v))
    ]
    var_list_new = [
        var - tf.realdiv(lr * mm, (tf.sqrt(vv) + epsilon))
        for var, mm, vv in list(zip(var_list, velocity_m_new, velocity_v_new))
    ]
    return var_list_new, velocity_m_new, velocity_v_new


def update(var_list, new_vars, global_step=None):
    """Update variables.

    Args:
        var_list: List of trainable variables.
        new_vars: List of trainable variables of the next training step.
        global_step: Counter of global step, adds one after variable updates.

    Returns:
        train_op: A TensorFlow op for applying gradient descent with momentum.
    """
    assert len(var_list) == len(new_vars)
    var_op = [
        tf.assign(x, x_new) for x, x_new in list(zip(var_list, new_vars))
    ]
    with tf.control_dependencies(var_op):
        if global_step is None:
            return tf.no_op()
        else:
            return tf.assign_add(global_step,
                                 tf.constant(1, dtype=global_step.dtype))


def lr_decay(init_lr, decay, time_const, global_step):
    """Gets a decayed learning rate using inverse decay rule.

    Args:
        init_lr: Initial learning rate at step 0.
        decay: Decay exponent.
        time_const: Time constant of the inverse decay rule.
        global_step: Time step.

    Returns:
        lr: Learning rate at the current time step.
    """
    decay = tf.constant(decay)
    decay_step = tf.cast(global_step, tf.float32)
    lr = tf.realdiv(init_lr,
                    tf.pow(1.0 + tf.realdiv(decay_step, decay_const),
                           self._decay_exp))
    return lr


class Optimizer(object):
    def __init__(self, hyperparams_dict, dtype=tf.float32):
        """Initializes an optimizer with a list of hyperparameters."""
        self._grads = None
        self._accumulators = None
        self._new_accumulators = None
        self._hyperparams = {}
        self._assign_ops = {}
        self._assign_ph = {}
        self._dtype = dtype
        for name, hp in hyperparams_dict.items():
            _var = tf.get_variable(
                name, [],
                dtype=dtype,
                initializer=tf.constant_initializer(hp),
                trainable=False)
            self._hyperparams[name] = _var
            self._assign_ph[name] = tf.placeholder(
                dtype, [], name=name + '_placeholder')
            self._assign_ops[name] = tf.assign(_var, self._assign_ph[name])

    def apply_gradients(self, grads_and_vars, global_step=None):
        """Applies gradients

        Args:
            grads_and_vars: List of tuples of the gradients and the variables.
            global_step: Tensor that records the global step. Optional.

        Returns:
            train_op: A TensorFlow op that applies the gradients to the variables.
        """
        raise NotImplemented()

    def minimize(self, cost, var_list=None, global_step=None,
                 gate_gradients=1):
        """Minimizes a cost function.

        Args:
            cost: Cost function to minimize.
            var_list: List of trainable variables, default tf.trainable_variables().
            global_step: Global step counter.
            gate_gradients: Whether to allow concurrency in calculating the gradients.

        Returns:
            train_op: A TensorFlow op that applies the gradients to the variables.
        """
        raise NotImplemented()

    @property
    def dtype(self):
        return self._dtype

    @property
    def grads(self):
        return self._grads

    @property
    def hyperparams(self):
        return self._hyperparams

    @property
    def accumulators(self):
        return self._accumulators

    @property
    def new_accumulators(self):
        return self._new_accumulators

    def assign_hyperparam(self, sess, hp_name, val):
        sess.run(
            self._assign_ops[hp_name],
            feed_dict={
                self._assign_ph[hp_name]: val
            })


class GradientDescentOptimizer(Optimizer):
    def __init__(self, lr, dtype=tf.float32):
        """Gradient descent optimizer.

        Args:
            lr: Float. Learning rate.
            dtype: Data type, default tf.float32.
        """
        super(GradientDescentOptimizer, self).__init__({'lr': lr}, dtype=dtype)

    def apply_gradients(self, grads_and_vars, global_step=None):
        """See above in class Optimizer."""
        grads = [g for g, v in grads_and_vars]
        var_list = [v for g, v in grads_and_vars]
        var_list_new = gradient_descent(grads, var_list, self._lr)
        train_op = update(var_list, var_list_new, global_step=global_step)
        self._var_list_new = var_list_new
        self._accumulators = {'w': var_list}
        self._new_accumulators = {'w': var_list_new}
        return train_op

    def minimize(self, cost, var_list=None, global_step=None,
                 gate_gradients=1):
        """See above in class Optimizer."""
        if var_list is None:
            var_list = tf.trainable_variables()
        self._var_list = var_list
        grads = tf.gradients(cost, var_list, gate_gradients=gate_gradients)
        self._grads = grads
        self._lr = self.hyperparams['lr']
        train_op = self.apply_gradients(
            list(zip(grads, var_list)), global_step=global_step)
        return train_op

    @property
    def var_list(self):
        return self._var_list

    @property
    def var_list_new(self):
        return self._var_list_new


class MomentumOptimizer(Optimizer):
    def __init__(self,
                 lr,
                 momentum=0.9,
                 effective_lr=False,
                 negative_momentum=False,
                 dtype=tf.float32):
        """Momentum Optimizer.

        Args:
            lr: Float. Learning rate.
            momentum: Float. Momentum, default 0.9.
            dtype: Data type, default tf.float32.
        """
        self._effective_lr = effective_lr
        self._negative_momentum = negative_momentum
        super(MomentumOptimizer, self).__init__(
            {
                'lr': lr,
                'mom': momentum
            }, dtype=dtype)

    def apply_gradients(self, grads_and_vars, global_step=None):
        """See above in class Optimizer."""
        velocity = create_velocity_variables(self._var_list)
        lr_ = self._lr
        mom_ = self._mom
        grads = [g for g, v in grads_and_vars]
        var_list = [v for g, v in grads_and_vars]
        var_list_new, velocity_new = gradient_descent_momentum(
            grads, velocity, var_list, lr_, mom_)
        train_op = update(
            var_list + velocity,
            var_list_new + velocity_new,
            global_step=global_step)
        self._var_list_new = var_list_new
        self._velocity = velocity
        self._velocity_new = velocity_new
        self._accumulators = {'w': var_list, 'v': velocity}
        self._new_accumulators = {'w': var_list_new, 'v': velocity_new}
        return train_op

    def reparameterize(self, lr, mom):
        """Re-parameterization of learning rate and momentum.

        Args:
            lr: Float. Learning rate value stored in hyperparameter variable.
            mom: Float. Momentum value stored in hyperparameter variable.

        Returns:
            lr_: If `effective_lr` is on, then it will return lr * (1 - mom), otherwise the  same
                learning rate value as the input.
            mom_: If `negative_momentum` is on, the it will return (1 - mom), otherwise, the same
                momentum value as the input.
        """
        if self._negative_momentum:
            mom_ = 1 - mom
        else:
            mom_ = mom
        if self._effective_lr:
            lr_ = lr * (1 - mom_)
        else:
            lr_ = lr
        return lr_, mom_

    def minimize(self, cost, var_list=None, global_step=None,
                 gate_gradients=1):
        """See above in class Optimizer."""
        if var_list is None:
            var_list = tf.trainable_variables()
        self._var_list = var_list
        grads = tf.gradients(cost, var_list, gate_gradients=gate_gradients)
        self._lr, self._mom = self.reparameterize(self.hyperparams['lr'],
                                                  self.hyperparams['mom'])
        grads = tf.gradients(cost, var_list, gate_gradients=True)
        self._grads = grads
        return self.apply_gradients(
            list(zip(grads, var_list)), global_step=global_step)

    @property
    def var_list(self):
        return self._var_list

    @property
    def var_list_new(self):
        return self._var_list_new

    @property
    def velocity(self):
        return self._velocity

    @property
    def velocity_new(self):
        return self._velocity_new


class MomentumInvDecayOptimizer(MomentumOptimizer):
    def __init__(self,
                 lr,
                 momentum=0.9,
                 decay=0.1,
                 time_const=5000.0,
                 effective_lr=False,
                 negative_momentum=False,
                 dtype=tf.float32):
        """Momentum Optimizer with inverse learning rate decay.

        Args:
            lr: Float. Learning rate.
            momentum: Float. Momentum, default 0.9.
            dtype: Data type, default tf.float32.
        """
        self._effective_lr = effective_lr
        self._negative_momentum = negative_momentum
        self._time_const = time_const
        Optimizer.__init__(
            self, {'lr': lr,
                   'mom': momentum,
                   'decay': decay}, dtype=dtype)

    def minimize(self, cost, var_list=None, global_step=None,
                 gate_gradients=1):
        """See above in class Optimizer."""
        if var_list is None:
            var_list = tf.trainable_variables()
        self._var_list = var_list

        if global_step is None:
            global_step = tf.get_variable(
                'global_step', [],
                dtype=tf.int64,
                initializer=tf.constant_initializer(0, dtype=tf.int64),
                trainable=False)

        grads = tf.gradients(cost, var_list, gate_gradients=gate_gradients)
        self._grads = grads

        self._lr, self._mom = self.reparameterize(self.hyperparams['lr'],
                                                  self.hyperparams['mom'])

        # Learning rate decay.
        decay = self.hyperparams['decay']
        t = tf.cast(global_step, self.dtype)
        time_const_f = tf.constant(self._time_const, dtype=self.dtype)
        self._lr = self._lr * tf.pow(1.0 + tf.realdiv(t, time_const_f), -decay)

        grads = tf.gradients(cost, var_list, gate_gradients=True)
        return self.apply_gradients(
            list(zip(grads, var_list)), global_step=global_step)

    @property
    def var_list(self):
        return self._var_list

    @property
    def var_list_new(self):
        return self._var_list_new

    @property
    def velocity(self):
        return self._velocity

    @property
    def velocity_new(self):
        return self._velocity_new


class AdamOptimizer(Optimizer):
    def __init__(self,
                 lr,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 negative_momentum=False,
                 dtype=tf.float32):
        """Momentum Optimizer.

        Args:
            lr: Float. Learning rate.
            momentum: Float. Momentum, default 0.9.
            dtype: Data type, default tf.float32.
        """
        self._epsilon = epsilon
        self._negative_momentum = negative_momentum
        super(AdamOptimizer, self).__init__(
            {
                'lr': lr,
                'beta1': beta1,
                'beta2': beta2
            }, dtype=dtype)

    def apply_gradients(self, grads_and_vars, global_step=None):
        """See above in class Optimizer."""
        velocity_m = create_velocity_variables(
            self._var_list, name='momentum_m')
        velocity_v = create_velocity_variables(
            self._var_list, name='momentum_v')
        lr_ = self._lr
        beta1_ = self._beta1
        beta2_ = self._beta2
        grads = [g for g, v in grads_and_vars]
        var_list = [v for g, v in grads_and_vars]
        var_list_new, velocity_m_new, velocity_v_new = adam(
            grads, velocity_m, velocity_v, var_list, lr_, beta1_, beta2_,
            self._epsilon)
        train_op = update(
            var_list + velocity_m + velocity_v,
            var_list_new + velocity_m_new + velocity_v_new,
            global_step=global_step)
        self._var_list_new = var_list_new
        self._velocity_m = velocity_m
        self._velocity_m_new = velocity_m_new
        self._velocity_v = velocity_v
        self._velocity_v_new = velocity_v_new
        self._accumulators = {'w': var_list, 'm': velocity_m, 'v': velocity_v}
        self._new_accumulators = {
            'w': var_list_new,
            'm': velocity_m_new,
            'v': velocity_v_new
        }
        return train_op

    def reparameterize(self, beta1, beta2):
        """Re-parameterization of learning rate and momentum.

        Args:
            beta1: Float. Learning rate value stored in hyperparameter variable.
            beta1: Float. Momentum value stored in hyperparameter variable.

        Returns:
            beta1_: If `negative_momentum` is on, the it will return (1 - beta1), otherwise, the
                same momentum value as the input.
            beta2_: If `negative_momentum` is on, the it will return (1 - beta2), otherwise, the
                same momentum value as the input.
        """
        if self._negative_momentum:
            beta1_ = 1 - beta1
            beta2_ = 1 - beta2
        else:
            beta1_ = beta1
            beta2_ = beta2
        return beta1_, beta2_

    def minimize(self, cost, var_list=None, global_step=None,
                 gate_gradients=1):
        """See above in class Optimizer."""
        if var_list is None:
            var_list = tf.trainable_variables()
        self._var_list = var_list

        if global_step is None:
            global_step = tf.get_variable(
                'global_step', [],
                dtype=tf.int64,
                initializer=tf.constant_initializer(0, dtype=tf.int64),
                trainable=False)

        grads = tf.gradients(cost, var_list, gate_gradients=gate_gradients)
        self._beta1, self._beta2 = self.reparameterize(
            self.hyperparams['beta1'], self.hyperparams['beta2'])
        t = tf.cast(global_step, self.dtype) + 1.0
        ratio = tf.realdiv(
            tf.sqrt(1.0 - tf.pow(self._beta2, t)),
            (1.0 - tf.pow(self._beta1, t)))
        self._lr = self.hyperparams['lr'] * ratio
        grads = tf.gradients(cost, var_list, gate_gradients=True)
        self._grads = grads
        return self.apply_gradients(
            list(zip(grads, var_list)), global_step=global_step)

    @property
    def var_list(self):
        return self._var_list

    @property
    def var_list_new(self):
        return self._var_list_new

    @property
    def velocity(self):
        return self._velocity

    @property
    def velocity_new(self):
        return self._velocity_new


class LogOptimizer(tf.train.Optimizer):
    """A optimizer wrapper that optimizes on log space."""

    def __init__(self, base_optimizer):
        """Initilialize log space optimizer.

        Args:
            base_optimizer: A regular tf.train.Optimizer object.
        """
        self._base_optimizer = base_optimizer

    def clip(self, v, clipv):
        """Clips a variable.

        Args:
            v: A variable.
            clipv: A tuple of min and max values to clip.
        """
        if clipv is None:
            return v
        minv, maxv = clipv
        return tf.maximum(tf.minimum(v, maxv), minv)

    def apply_gradients(self,
                        grads_and_vars,
                        global_step=None,
                        name=None,
                        clip_gradients=None,
                        clip_values=None):
        """Applies gradients to variables, on log space.

        Args:
            grads_and_vars: A list of tuples of gradients and variables.
            global_step: Time step t, will be incremented with train_op.
            name: Name of the returned operation.
            clip_values: Clip of the values of the optimized variables.

        Returns:
            train_op: Assigns the new values to the variables.
        """
        grads_and_vars = list(
            filter(lambda x: x[0] is not None, grads_and_vars))
        if clip_gradients is not None:
            new_grads = [
                self.clip(gv[0], _clip_g)
                for gv, _clip_g in list(zip(grads_and_vars, clip_gradients))
            ]
        else:
            new_grads = [gg for gg, vv in grads_and_vars]

        old_v = [vv for gg, vv in grads_and_vars]
        tmp_v = [
            tf.get_variable(
                vv.name.split(':')[0] + '_log_adam',
                vv.get_shape(),
                trainable=False,
                dtype=vv.dtype) for vv in old_v
        ]
        var_to_opt = [
            tf.assign(tv, tf.log(vv)) for tv, vv in list(zip(tmp_v, old_v))
        ]
        var_grad = [vv * gg for gg, vv in list(zip(new_grads, old_v))]
        with tf.control_dependencies(var_to_opt):
            op = self._base_optimizer.apply_gradients(
                list(zip(var_grad, tmp_v)))
        with tf.control_dependencies([op]):
            new_v = [tf.exp(tv) for tv in tmp_v]
            if clip_values is not None:
                new_v = [
                    self.clip(vv, _clip_v)
                    for vv, _clip_v in list(zip(new_v, clip_values))
                ]
            train_op = tf.group(*[
                tf.assign(_old_v, _new_v)
                for _old_v, _new_v in list(zip(old_v, new_v))
            ])
        return train_op
