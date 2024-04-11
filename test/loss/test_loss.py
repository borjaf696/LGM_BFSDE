import pytest
import tensorflow as tf
from scripts.losses.losses import Losses


@pytest.fixture
def random_tensor_1():
    return tf.random.normal([999, 10], mean=0, stddev=5)


@pytest.fixture
def random_tensor_2():
    return tf.random.normal([999, 10], mean=0, stddev=2)


@pytest.fixture
def tensor_1():
    return tf.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.float32)


@pytest.fixture
def tensor_1b():
    return tf.constant([3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32)


@pytest.fixture
def tensor_1c():
    return tf.constant([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1], dtype=tf.float32)


@pytest.fixture
def L1():
    return tf.math.abs


@pytest.fixture
def L2():
    return tf.math.squared_difference


# Test normalized loss function
def test_normalize_loss_not_nans(random_tensor_1, random_tensor_2, L1, L2):
    result_loss = Losses.get_normalized_loss(random_tensor_1, random_tensor_2, L1, L2)
    nans = tf.math.is_nan(result_loss)
    num_nans = tf.reduce_sum(tf.cast(nans, tf.int32))

    assert num_nans == 0


def test_normalize_loss_higher_than_1(tensor_1, tensor_1b, L1, L2):
    result_loss = Losses.get_normalized_loss(tensor_1, tensor_1b, L1, L2)

    agg_result = tf.reduce_mean(result_loss)

    assert (agg_result - 4.0) < 1e-5


def test_normalize_loss_higher_smaller_than_1(tensor_1, tensor_1c, L1, L2):
    result_loss = Losses.get_normalized_loss(tensor_1, tensor_1c, L1, L2)

    agg_result = tf.reduce_mean(result_loss)

    assert (agg_result - 0.1) < 1e-5


# Test loss function
def test_loss_not_nans(random_tensor_1, random_tensor_2, L1, L2):
    result_loss = Losses.get_loss(random_tensor_1, random_tensor_2, L1, L2)
    nans = tf.math.is_nan(result_loss)
    num_nans = tf.reduce_sum(tf.cast(nans, tf.int32))

    assert num_nans == 0


def test_loss_higher_than_1(tensor_1, tensor_1b, L1, L2):
    result_loss = Losses.get_loss(tensor_1, tensor_1b, L1, L2)

    agg_result = tf.reduce_mean(result_loss)

    assert (agg_result - 4.0) < 1e-5


def test_loss_higher_smaller_than_1(tensor_1, tensor_1c, L1, L2):
    result_loss = Losses.get_loss(tensor_1, tensor_1c, L1, L2)

    agg_result = tf.reduce_mean(result_loss)

    assert (agg_result - 0.1) < 1e-5
