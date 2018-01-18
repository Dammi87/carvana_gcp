import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import src.lib.tfops as tfops

def simple(features, mode, scope='simple_network'):
    """Returns a simple network architecture.

    conv[5,5,32] -> Dense -> Dropout -> deconv[5,5,2]

    Parameters
    features : Tensor
        4D Tensor where the first dimension is the batch size, then height, width
        and channels
    mode : tensorflow.python.estimator.model_fn.ModeKeys
        Class that contains the current mode
    scope : str, optional
        The scope to use for this architecture

    Returns
    -------
    Tensor op
        Return the final tensor operation, or logits, from the network
    """
    with tf.variable_scope(scope):
        is_training = mode == Modes.TRAIN

        # Input Layer
        net = [features]
        # Convolutional Layer #1
        net.append(tf.layers.conv2d(inputs=net[-1],
                                    filters=32,
                                    kernel_size=[5, 5],
                                    padding="same",
                                    name="conv_1_1",
                                    activation=tf.nn.relu))

        # Fully connected layer
        net.append(tf.layers.dense(inputs=net[-1], units=16, activation=tf.nn.relu))

        # Dropout
        net.append(tf.layers.dropout(inputs=net[-1], rate=0.4, training=is_training))
        # Deconv
        net.append(tfops.deconv2d_resize(inputs=net[-1],
                                         filters=2,
                                         kernel_size=[5, 5],
                                         padding="same",
                                         activation=tf.nn.relu))

        return net[-1]
