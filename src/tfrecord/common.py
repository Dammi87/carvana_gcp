import tensorflow as tf


def _int64_feature(value):
    """Return value as a int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Return value as a byte feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_img_img(tfrecord_path, ftr_chunks, lbl_chunks, get_tfexample, resize=self._img_resize):
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
        with tf.Session('') as sess:
            for irecord, features, labels in zip(tfrecord_path, ftr_chunks, lbl_chunks):
                with tf.python_io.TFRecordWriter(irecord) as tfrecord_writer:
                    for iftr, ilbl in zip(features, labels):
                        example = get_tfexample(iftr, ilbl)
                        tfrecord_writer.write(example.SerializeToString())

