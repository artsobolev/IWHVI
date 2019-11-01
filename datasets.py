import os
import collections
import urllib

import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.io


_seed = 1234
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

MNIST_BINARIZED_URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist'
OMNIGLOT_URL = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'


class DataSet(object):
    def __init__(self, images, labels, shuffle=True):
        """
            Construct a DataSet.
        """
        assert images.shape[0] == labels.shape[0], 'images.shape: %s labels.shape: %s' % (images.shape, labels.shape)
        assert len(images.shape) == 2

        self._num_examples = images.shape[0]
        self._shuffle = shuffle
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._shuffle_next = shuffle

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

        if self._shuffle_next:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
            self._shuffle_next = False

        start = self._index_in_epoch
        batch_size = min(batch_size, self._num_examples - start)
        end = self._index_in_epoch = start + batch_size

        # Go to the next epoch
        if start + batch_size == self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            self._index_in_epoch = 0
            self._shuffle_next = self._shuffle

        return self._images[start:end], self._labels[start:end]


def get_static_mnist(datasets_dir, validation_size=5000):
    def lines_to_np_array(lines):
        return np.array([list(map(int, line.split())) for line in lines], dtype=np.uint8) * 255

    mnist_dir = os.path.join(datasets_dir, 'BinaryMNIST')

    data = {}
    for split in ['train', 'valid', 'test']:
        mnist_split_path = os.path.join(mnist_dir, 'binarized_mnist_%s.amat' % split)

        if not os.path.exists(mnist_split_path):
            os.makedirs(mnist_dir, exist_ok=True)
            urllib.request.urlretrieve(MNIST_BINARIZED_URL + '/binarized_mnist_%s.amat' % split, mnist_split_path)

        with open(mnist_split_path) as f:
            data[split] = lines_to_np_array(f.readlines())

    train_data = np.concatenate((data['valid'], data['train']))
    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]
    test_data = data['test']

    return Datasets(
        train=DataSet(train_data, train_data),
        validation=DataSet(validation_data, validation_data),
        test=DataSet(test_data, test_data)
    )


def get_dynamic_mnist(datasets_dir):
    from tensorflow.examples.tutorials.mnist import input_data
    orig_dataset = input_data.read_data_sets(os.path.join(datasets_dir, 'MNIST'), one_hot=False, dtype=tf.uint8)

    return Datasets(
        train=DataSet(orig_dataset.train.images, orig_dataset.train.images),
        validation=DataSet(orig_dataset.validation.images, orig_dataset.validation.images),
        test=DataSet(orig_dataset.test.images, orig_dataset.test.images)
    )


def get_intraclass_paired_mnist(datasets_dir):
    from tensorflow.examples.tutorials.mnist import input_data
    orig_dataset = input_data.read_data_sets(os.path.join(datasets_dir, 'MNIST'), one_hot=False, dtype=tf.uint8)

    def pair(dataset):
        xs = []
        ys = []
        for digit in range(10):
            mask = dataset.labels == digit
            images = dataset.images[mask]

            for idx in range(0, len(images) - 1, 2):
                x1, x2 = images[idx:idx+2]

                for a in [x1, x2]:
                    for b in [x1, x2]:
                        xs.append(a)
                        ys.append(b)

        return np.array(xs), np.array(ys)

    return Datasets(
        train=DataSet(*pair(orig_dataset.train)),
        validation=DataSet(*pair(orig_dataset.validation)),
        test=DataSet(*pair(orig_dataset.test))
    )


def get_class_paired_mnist(datasets_dir):
    from tensorflow.examples.tutorials.mnist import input_data
    orig_dataset = input_data.read_data_sets(os.path.join(datasets_dir, 'MNIST'), one_hot=False, dtype=tf.uint8)

    def pair(dataset):
        xs = []
        ys = []
        for digit1 in range(0, 10, 2):
            digit2 = digit1 + 1
            images1 = dataset.images[dataset.labels == digit1]
            images2 = dataset.images[dataset.labels == digit2]

            for x1, x2 in zip(images1, images2):
                for a in [x1, x2]:
                    for b in [x1, x2]:
                        xs.append(a)
                        ys.append(b)

        return np.array(xs), np.array(ys)

    return Datasets(
        train=DataSet(*pair(orig_dataset.train)),
        validation=DataSet(*pair(orig_dataset.validation)),
        test=DataSet(*pair(orig_dataset.test))
    )


def get_omniglot(datasets_dir, validation_size=5000):
    def reshape_omni(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='fortran')

    omniglot_dir = os.path.join(datasets_dir, 'OMNIGLOT')
    omniglot_path = os.path.join(omniglot_dir, 'chardata.mat')
    if not os.path.isfile(omniglot_path):
        os.makedirs(omniglot_dir, exist_ok=True)
        urllib.request.urlretrieve(OMNIGLOT_URL, omniglot_path)

    omni_raw = sp.io.loadmat(omniglot_path)
    train_data = reshape_omni(omni_raw['data'].T * 255).astype(np.uint8)
    test_data = reshape_omni(omni_raw['testdata'].T * 255).astype(np.uint8)
    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]

    return Datasets(
        train=DataSet(train_data, train_data),
        validation=DataSet(validation_data, validation_data),
        test=DataSet(test_data, test_data)
    )
