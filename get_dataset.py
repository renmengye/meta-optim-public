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
"""Gets a dataset.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
import mnist
import cifar10
import os


def get_dataset(dataset_name, seed=0, test=False):
    if dataset_name == "mnist":
        data_folder = './data/mnist'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        if not test:
            dataset = mnist.read_data_sets("./data/mnist", seed=seed).train
        else:
            dataset = mnist.read_data_sets("./data/mnist", seed=seed).test
    elif dataset_name == "cifar-10":
        if not test:
            dataset = cifar10.read_data_sets("./data/cifar-10", seed=seed).train
        else:
            dataset = cifar10.read_data_sets("./data/cifar-10", seed=seed).test
    else:
        raise Exception("Not implemented.")
    return dataset
