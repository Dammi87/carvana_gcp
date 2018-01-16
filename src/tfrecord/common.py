import tensorflow as tf

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""
  
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
  
    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]
  
    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _int64_feature(value):
    """Return value as a int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Return value as a byte feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_img_img(tfrecord_path, ftr_chunks, lbl_chunks, get_tfexample):
    """Convert the lists given into tfrecords.

    Parameters
    ----------
        tfrecord_path: list
            list of tfrecord paths to use for conversions
        ftr_chunks: list
            List of lists of paths to feature images per tf_record
        lbl_chunks: list
            List of lists of paths to feature images per tf_record
        get_tfexample: method
            A method that takes in the feature and label and returns the tf example
            feature that is desired
    """

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for irecord, features, labels in zip(tfrecord_path, ftr_chunks, lbl_chunks):
                with tf.python_io.TFRecordWriter(irecord) as tfrecord_writer:
                    for iftr, ilbl in zip(features, labels):
                        example = get_tfexample(iftr, ilbl)
                        tfrecord_writer.write(example.SerializeToString())

