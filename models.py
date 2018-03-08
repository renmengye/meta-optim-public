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
Simple models.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from logger import get as get_logger
from optimizers import AdamOptimizer
from optimizers import GradientDescentOptimizer
from optimizers import MomentumInvDecayOptimizer
from optimizers import MomentumOptimizer

log = get_logger()


def mlp(x, layer_sizes, init_std, seed=0, dtype=tf.float32, wd=0.0, dropout=None, act_fn=None):
    """A simple MLP model."""
    if wd > 0.0:
        _reg = lambda x: tf.multiply(tf.nn.l2_loss(x), wd)
    else:
        _reg = None
    w = [
        tf.get_variable(
            "w{}".format(ii),
            shape=[layer_sizes[ii], layer_sizes[ii + 1]],
            dtype=dtype,
            initializer=tf.random_normal_initializer(stddev=init_std[ii], seed=seed),
            regularizer=_reg) for ii in range(len(layer_sizes) - 1)
    ]
    b = [
        tf.get_variable(
            "b{}".format(ii),
            shape=[layer_sizes[ii + 1]],
            dtype=dtype,
            initializer=tf.constant_initializer(0.0)) for ii in range(len(layer_sizes) - 1)
    ]
    h = x
    for ii in range(len(layer_sizes) - 1):
        log.info('Fully connected {} {}'.format(layer_sizes[ii], layer_sizes[ii + 1]))
        h = tf.matmul(h, w[ii]) + b[ii]

        if ii < len(layer_sizes) - 2:
            if dropout is not None:
                if dropout[ii]:
                    log.info('Dropout')
                    h = tf.nn.dropout(h, keep_prob=0.5)
            if act_fn is not None:
                log.info('Activation {}'.format(act_fn[ii]))
                h = act_fn[ii](h)
    return h


def cnn(x,
        conv_filter_sizes,
        conv_init_std,
        mlp_layer_sizes,
        mlp_init_std,
        seed=0,
        wd=0.0,
        dtype=tf.float32):
    """A simple CNN model."""
    if wd > 0.0:
        _reg = lambda x: tf.multiply(tf.nn.l2_loss(x), wd)
    else:
        _reg = None
    conv_w = [
        tf.get_variable(
            "conv_w{}".format(ii),
            shape=conv_filter_sizes[ii],
            dtype=dtype,
            initializer=tf.random_normal_initializer(stddev=conv_init_std[ii], seed=seed),
            regularizer=_reg) for ii in range(len(conv_filter_sizes))
    ]
    conv_b = [
        tf.get_variable(
            "conv_b{}".format(ii),
            shape=[conv_filter_sizes[ii][-1]],
            dtype=dtype,
            initializer=tf.constant_initializer(0.0)) for ii in range(len(conv_filter_sizes))
    ]
    h = x
    for ii in range(len(conv_filter_sizes)):
        h = tf.nn.conv2d(h, conv_w[ii], strides=[1, 1, 1, 1], padding="SAME") + conv_b[ii]
        h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        h = tf.nn.relu(h)
    mlp_w = [
        tf.get_variable(
            "mlp_w{}".format(ii),
            shape=[mlp_layer_sizes[ii], mlp_layer_sizes[ii + 1]],
            dtype=dtype,
            initializer=tf.random_normal_initializer(stddev=mlp_init_std[ii], seed=seed))
        for ii in range(len(mlp_layer_sizes) - 1)
    ]
    mlp_b = [
        tf.get_variable(
            "b{}".format(ii),
            shape=[mlp_layer_sizes[ii + 1]],
            dtype=dtype,
            initializer=tf.constant_initializer(0.0)) for ii in range(len(mlp_layer_sizes) - 1)
    ]
    h_shape = h.get_shape()
    h = tf.reshape(h, [-1, int(h_shape[1]) * int(h_shape[2]) * int(h_shape[3])])
    for ii in range(len(mlp_layer_sizes) - 1):
        h = tf.matmul(h, mlp_w[ii]) + mlp_b[ii]
        if ii < len(mlp_layer_sizes) - 2:
            h = tf.nn.relu(h)
    return h


def _variable_on_cpu(name, shape, initializer, dtype=tf.float32):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, dtype=tf.float32):
    """
    Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(
        name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype), dtype=dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def tf_cifar_cnn():
    """Build the CIFAR-10 model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Logits.
    """
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(
        norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', [192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear


# --------------------------------------------------------------------
# Network architecture parameters.


class MLPConfig(object):
    layer_sizes = [784, 100, 100, 10]
    init_std = [0.1, 0.1, 0.1]
    wd = 0e0
    dropout = None
    act_fn = [tf.nn.relu, tf.nn.relu]
    init_lr = 0.1
    momentum = 0.9
    lr_decay = 0.0
    effective_lr = False
    negative_momentum = False


def get_mnist_mlp_config(init_lr, momentum, decay=0.0, effective_lr=False, negative_momentum=False):
    """Return a standard MLP model for MNIST.

    Args:
        init_lr: Float. Initial learning rate.
        momentum: Float. Momentum term.

    Returns:
        config: A MLP config that has a standard network architecture.
    """
    config = MLPConfig()
    config.layer_sizes = [784, 100, 100, 10]
    config.init_std = [0.05, 0.05, 0.05]
    config.wd = 0e0
    config.dropout = None
    config.act_fn = [tf.nn.relu, tf.nn.relu]
    config.init_lr = init_lr
    config.momentum = momentum
    config.lr_decay = decay
    config.effective_lr = effective_lr
    config.negative_momentum = negative_momentum
    return config


class CNNConfig(object):
    conv_filter_sizes = [[5, 5, 3, 32], [5, 5, 32, 32], [5, 5, 32, 64]]
    mlp_layer_sizes = [1024, 100, 10]
    conv_init_std = [0.1, 0.1, 0.1]
    mlp_init_std = [0.1, 0.1]
    init_lr = 0.01
    momentum = 0.9
    lr_decay = 0.0
    effective_lr = False
    negative_momentum = False


def get_cifar_cnn_config(init_lr, momentum, decay=0.0, effective_lr=False, negative_momentum=False):
    """Return a standard CNN model for CIFAR.

    Args:
        init_lr: Float. Initial learning rate.
        momentum: Float. Momentum term.

    Returns:
        config: A CNN config that has a standard network architecture.
    """
    config = CNNConfig()
    config.conv_filter_sizes = [[5, 5, 3, 32], [5, 5, 32, 32], [5, 5, 32, 64]]
    config.mlp_layer_sizes = [1024, 100, 10]
    config.conv_init_std = [0.1, 0.1, 0.1]
    config.mlp_init_std = [0.1, 0.1]
    config.init_lr = init_lr
    config.momentum = momentum
    config.lr_decay = decay
    config.effective_lr = effective_lr
    config.negative_momentum = negative_momentum
    return config


def get_mnist_mlp_config(init_lr, momentum, decay=0.0, effective_lr=False, negative_momentum=False):
    """Return a standard MLP model for MNIST.

    Args:
        init_lr: Float. Initial learning rate.
        momentum: Float. Momentum term.

    Returns:
        config: A MLP config that has a standard network architecture.
    """
    config = MLPConfig()
    config.layer_sizes = [784, 100, 100, 10]
    config.init_std = [0.05, 0.05, 0.05]
    config.wd = 0e0
    config.dropout = None
    config.act_fn = [tf.nn.relu, tf.nn.relu]
    config.init_lr = init_lr
    config.momentum = momentum
    config.lr_decay = decay
    config.effective_lr = effective_lr
    config.negative_momentum = negative_momentum
    return config


# --------------------------------------------------------------------
# Network model base class.


class Model(object):
    def __init__(self, config, x, y, optimizer='momentum', dtype=tf.float32, training=True):
        self._config = config
        self._dtype = dtype
        h = self.build_inference(x)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        log.info(('Regularization', reg_losses))
        if len(reg_losses) > 0:
            wd_cost = tf.add_n(reg_losses)
            cost = xent + wd_cost
            print(reg_losses)
            log.fatal('')
        else:
            wd_cost = tf.constant(0.0)
            cost = xent
        self._cost = cost

        correct = tf.reduce_sum(tf.to_float(tf.equal(tf.argmax(h, 1), y)))
        acc = tf.realdiv(correct, tf.to_float(tf.shape(x)[0]))

        self._acc = acc
        self._correct = correct
        var_list = tf.trainable_variables()
        # print('Variables')
        # [print(vv.name) for vv in var_list]
        self._var_list = var_list
        self._x = x
        self._y = y

        if training:
            global_step = tf.get_variable(
                'global_step', [],
                dtype=tf.int64,
                initializer=tf.constant_initializer(0, dtype=tf.int64),
                trainable=False)
            if optimizer == 'momentum':
                opt = MomentumOptimizer(
                    config.init_lr,
                    config.momentum,
                    effective_lr=config.effective_lr,
                    negative_momentum=config.negative_momentum,
                    dtype=dtype)
            elif optimizer == 'momentum_inv_decay':
                opt = MomentumInvDecayOptimizer(
                    config.init_lr,
                    config.momentum,
                    config.lr_decay,
                    effective_lr=config.effective_lr,
                    negative_momentum=config.negative_momentum,
                    dtype=dtype)
            elif optimizer == 'gradient_descent':
                opt = GradientDescentOptimizer(config.init_lr, dtype=dtype)
            elif optimizer == 'adam':
                opt = AdamOptimizer(config.init_lr, dtype=dtype)
            else:
                raise ValueError('Unknown Optimizer')
            train_op = opt.minimize(cost, global_step=global_step)
            self._train_op = train_op
            self._optimizer = opt
            self._global_step = global_step
        else:
            self._optimizer = None

    @property
    def config(self):
        return self._config

    @property
    def dtype(self):
        return self._dtype

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def acc(self):
        return self._acc

    @property
    def correct(self):
        return self._correct

    @property
    def optimizer(self):
        return self._optimizer

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

    @property
    def grads(self):
        return self._grads

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_op(self):
        return self._train_op

    @property
    def cost(self):
        return self._cost

    def build_inference(self, x):
        raise NotImplemented()


class MLPModel(Model):
    def build_inference(self, x):
        config = self.config
        x_ = tf.reshape(x, [-1, 784])
        h = mlp(
            x_,
            layer_sizes=config.layer_sizes,
            init_std=config.init_std,
            wd=config.wd,
            dropout=config.dropout,
            dtype=self.dtype,
            act_fn=config.act_fn)
        return h


class CNNModel(Model):
    def build_inference(self, x):
        config = self.config
        x_ = tf.reshape(x, [-1, 32, 32, 3])
        h = cnn(
            x_,
            conv_filter_sizes=config.conv_filter_sizes,
            conv_init_std=config.conv_init_std,
            mlp_layer_sizes=config.mlp_layer_sizes,
            mlp_init_std=config.mlp_init_std,
            seed=0,
            wd=0.0,
            dtype=self.dtype)
        return h


def get_mnist_mlp_model(*args, **kwargs):
    """Builds a model for training MNIST MLP.

    Args:
        config: MLPConfig tuple object.
        x: 4D Tensor. Input images [N, H, W, C].
        y: 1D Tensor. Label integer indices.
        dtype: Data type, default tf.float32.
        training: Bool. Whether in training mode, default True.

    Returns:
        mlp: An MLP object.
    """
    return MLPModel(*args, **kwargs)


def get_cifar_cnn_model(*args, **kwargs):
    """Builds a model for training CIFAR CNN.

    Args:
        config: MLPConfig tuple object.
        x: 4D Tensor. Input images [N, H, W, C].
        y: 1D Tensor. Label integer indices.
        dtype: Data type, default tf.float32.
        training: Bool. Whether in training mode, default True.

    Returns:
        cnn: A CNN object.
    """
    return CNNModel(*args, **kwargs)
