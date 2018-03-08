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
Checkpointing utilities.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def build_checkpoint(var_list):
    with tf.variable_scope("checkpoint"):

        def get_var(x):
            return tf.get_variable(
                x.name.split(":")[0],
                x.get_shape(),
                x.dtype,
                tf.constant_initializer(0, dtype=x.dtype),
                trainable=False)

        ckpt = list(map(get_var, var_list))
    return ckpt


def read_checkpoint(ckpt, var_list, use_locking=False):
    return tf.group(*[tf.assign(ww, ck, use_locking=use_locking) for ck, ww in zip(ckpt, var_list)])


def write_checkpoint(ckpt, var_list, use_locking=False):
    return tf.group(*[tf.assign(ck, ww, use_locking=use_locking) for ck, ww in zip(ckpt, var_list)])
