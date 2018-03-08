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
"""CIFAR 10 dataset.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import print_function, division

import os
import numpy as np
import six
import sys
if sys.version.startswith("3"):
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
from subprocess import call
from dataset import DataSet

SOURCE_URL = "https://www.cs.toronto.edu/~kriz/"


def maybe_download(filename, work_directory):
    """Download the data from Alex"s website, unless it"s already here."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(os.path.join(work_directory, 'cifar-10-batches-py')):
        if not os.path.exists(filepath):
            print("Downloading CIFAR-10 from {}".format(SOURCE_URL + filename))
            filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
            statinfo = os.stat(filepath)
            print("Succesfully downloaded {} {} bytes".format(filename, statinfo.st_size))
        print("Unzipping data")
        call(["tar", "-xzvf", filepath, "-C", work_directory])
        print("Done")
    return filepath


def read_data_sets(data_folder, seed=0):
    train_img = []
    train_label = []
    test_img = []
    test_label = []
    filename = 'cifar-10-python.tar.gz'
    maybe_download(filename, data_folder)

    train_file_list = [
        "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"
    ]
    test_file_list = ["test_batch"]

    for i in six.moves.xrange(len(train_file_list)):
        tmp_dict = np.load(
            os.path.join(data_folder, 'cifar-10-batches-py', train_file_list[i]), encoding='latin1')
        train_img.append(tmp_dict["data"])
        train_label.append(tmp_dict["labels"])

    tmp_dict = np.load(
        os.path.join(data_folder, 'cifar-10-batches-py', test_file_list[0]), encoding='latin1')
    test_img.append(tmp_dict["data"])
    test_label.append(tmp_dict["labels"])

    train_img = np.concatenate(train_img)
    train_label = np.concatenate(train_label)
    test_img = np.concatenate(test_img)
    test_label = np.concatenate(test_label)

    train_img = np.reshape(train_img, [-1, 3, 32, 32])
    test_img = np.reshape(test_img, [-1, 3, 32, 32])

    # change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
    train_img = np.transpose(train_img, [0, 2, 3, 1])
    test_img = np.transpose(test_img, [0, 2, 3, 1])

    class DataSets(object):
        pass

    data_sets = DataSets()
    data_sets.train = DataSet(train_img, train_label, seed=seed)
    data_sets.test = DataSet(test_img, test_label, seed=seed)

    return data_sets
