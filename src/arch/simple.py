import tensorflow as tf
import src.lib.tfops as tfops

def model(features, mode, scope='simple_network'):
    with tf.variable_scope(scope):
        is_training = mode == ModeKeys.TRAIN

        # Input Layer
        net = [features]

        # Convolutional Layer #1
        net.append(tf.layers.conv2d(
          inputs=net[-1],
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          name="conv_1_1",
          activation=tf.nn.relu))

        # Fully connected layer
        tensor_shape = net[-1].get_shape().as_list()
        units = tensor_shape[1] * tensor_shape[2] * tensor_shape[3] 
        net.append(tf.layers.dense(inputs=net[-1], units=units))

        # Dropout
        net.append(tf.layers.dropout(inputs=net[-1], rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN))

        # Deconv
        net.append(tfops.deconv2d_resize(
          inputs=net[-1],
          filters=2,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu))

        return net[-1]

        
