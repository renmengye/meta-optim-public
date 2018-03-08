# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
"""The Dataset object.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np

import logger

log = logger.get()


class DataSet(object):
    def __init__(self, images, labels, seed=0):
        assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" %
                                                    (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._orig_images = images
        self._orig_labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        log.info('seed ' + str(seed))
        self._seed = seed
        self._rnd = np.random.RandomState(seed)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        if self._index_in_epoch + batch_size >= self.num_examples:
            self._index_in_epoch = 0
            # Finished epoch
            self._epochs_completed += 1

        if self._index_in_epoch == 0:
            perm = np.arange(self.num_examples)
            self._rnd.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = start
            assert batch_size <= self.num_examples
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        assert 0 <= start <= self.num_examples
        assert 0 <= end <= self.num_examples
        _im, _lab = self._images[start:end], self._labels[start:end]
        assert _im.shape[0] == batch_size
        assert _lab.shape[0] == batch_size
        return _im, _lab

    def reset(self):
        """Resets iterator."""
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._rnd = np.random.RandomState(self._seed)
        self._images = self._orig_images
        self._labels = self._orig_labels

    def reverse(self, num_batch, batch_size):
        self._index_in_epoch -= num_batch * batch_size
        self._index_in_epoch = self._index_in_epoch % self.num_examples
