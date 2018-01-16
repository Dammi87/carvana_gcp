import tensorflow as tf
import src.tfrecord.common as common
from src.lib import fileops, imgops, listops
import os
from math import ceil

def _get_img_img_example(feature_path, label_path):
    """Create the feature/label pair tf example"""
    ftr, ftr_shape = imgops.load_image(feature_path)
    lbl, lbl_shape = imgops.load_image(label_path)
        
    feature = {
        'feature/img': common._bytes_feature(tf.compat.as_bytes(ftr.tostring())),
        'feature/height': common._int64_feature(ftr_shape[0]),
        'feature/width': common._int64_feature(ftr_shape[1]),
        'feature/channels': common._int64_feature(ftr_shape[2]),
        'label/img': common._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
        'label/height': common._int64_feature(lbl_shape[0]),
        'label/width': common._int64_feature(lbl_shape[1]),
        'label/channels': common._int64_feature(lbl_shape[2])
        }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _img_img_read_and_decode(record):
    """Decdoes the above example"""

    feature = {
        'feature/img': tf.FixedLenFeature([], tf.string),
        'feature/height': tf.FixedLenFeature([], tf.int64),
        'feature/width': tf.FixedLenFeature([], tf.int64),
        'feature/channels': tf.FixedLenFeature([], tf.int64),
        'label/img': tf.FixedLenFeature([], tf.string),
        'label/height': tf.FixedLenFeature([], tf.int64),
        'label/width': tf.FixedLenFeature([], tf.int64),
        'label/channels': tf.FixedLenFeature([], tf.int64)
        }

    parsed = tf.parse_single_example(record, feature)

    # Get the sizes
    ftr_height = tf.cast(parsed['feature/height'], tf.int32)
    ftr_width = tf.cast(parsed['feature/width'], tf.int32)
    ftr_channel = tf.cast(parsed['feature/channels'], tf.int32)
    lbl_height = tf.cast(parsed['label/height'], tf.int32)
    lbl_width = tf.cast(parsed['label/width'], tf.int32)
    lbl_channel = tf.cast(parsed['label/channels'], tf.int32)

    # shape of image and annotation
    ftr_shape = tf.stack([ftr_height, ftr_width, ftr_channel])
    lbl_shape = tf.stack([lbl_height, lbl_width, lbl_channel])

    # read, decode and normalize image
    feature_img = tf.decode_raw(parsed['feature/img'], tf.uint8)
    feature_img = tf.cast(feature_img, tf.float32) * (1. / 255) - 0.5
    feature_img = tf.reshape(feature_img, ftr_shape)
    label_img = tf.decode_raw(parsed['label/img'], tf.uint8)
    label_img = tf.cast(label_img, tf.int32)
    label_img = tf.reshape(label_img, lbl_shape)

    return feature_img, label_img

class ImgImgParser():
    def __init__(self, output_path, feature_folder, label_folder, split=[0.8, 0.1, 0.1]):
        self._feature_folder = feature_folder
        self._label_folder = label_folder
        self._split = split
        self._output_path = output_path

    def _get_image_paths(self, path):
        return fileops.get_all_files(path)

    def create_records(self, shards=1):
        """Create n_shards tfrecords at output_path."""
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

        feature_path = self._get_image_paths(self._feature_folder)
        label_path = self._get_image_paths(self._label_folder)

        # Get the data split
        split = fileops.split_lists([feature_path, label_path], *self._split)
        split_names = ['train', 'val', 'test']

        # Try to split up the "shards" between the dataset splits
        test_shards = ceil(shards * self._split[2])
        val_shards = ceil(shards * self._split[1])
        train_shards = ceil(shards * self._split[0])
        n_shards = [train_shards, val_shards, test_shards]

        # Loop through
        for _name, (feature_paths, label_paths), _n_shard in zip(split_names, split, n_shards):
            # Create the tfrecord name for each shard
            tf_names = [os.path.join(self._output_path, '%s_shard_%d.tfrecords' % (_name, i)) for i in range(_n_shard)]
            sharded_features = listops.chunks(feature_paths, _n_shard)
            sharded_labels = listops.chunks(label_paths, _n_shard)
            common.convert_img_img(tf_names, sharded_features, sharded_labels, _get_img_img_example)

    def get_input_feeder(self, data_type, batch_size, buffer_size, epochs=None):
        assert data_type in ['train', 'val', 'test']

        # Get all records of the desired type
        tfrecords = fileops.get_all_files_containing(self._output_path, '%s_shard_'% data_type)

        def feeder():
            dataset = tf.data.TFRecordDataset(tfrecords)

            def parser(record):
                features, labels = _img_img_read_and_decode(record)
                return features, labels

            dataset = dataset.map(parser)
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)
            if epochs:
                dataset = dataset.repeat(epochs)
            iterator = dataset.make_one_shot_iterator()

            return iterator.get_next()

        return feeder