import tensorflow as tf

def deconv2d_resize(inputs, filters, kernel_size, padding='SAME', strides=(1, 1), reuse=None, name=None, activation=None):
    """Resize input using nearest neighbor then apply convolution."""
    shape = inputs.get_shape().as_list()
    height = shape[1] * strides[0]
    width = shape[2] * strides[1]
    resized = tf.image.resize_nearest_neighbor(inputs, [height, width])

    return tf.layers.conv2d(inputs, filters,
                            kernel_size=kernel_size,
                            padding='SAME',
                            strides=strides,
                            reuse=reuse,
                            name=name,
                            activation=activation)