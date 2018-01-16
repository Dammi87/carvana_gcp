import tensorflow as tf
import src.tfrecord.common as common
from src.lib import fileops, imgops, listops
import os
from math import ceil

def _get_img_img_example(feature_path, label_path):
    """Create the feature/label pair tf example"""
    ftr = imgops.load_image(feature_path)
    lbl = imgops.load_image(label_path)
    feature = {
        'feature/img': common._bytes_feature(tf.compat.as_bytes(ftr.tostring())),
        'feature/height': common._int64_feature(ftr.shape[0]),
        'feature/width': common._int64_feature(ftr.shape[1]),
        'feature/channels': common._int64_feature(ftr.shape[2]),
        'label/img': common._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
        'label/height': common._int64_feature(lbl.shape[0]),
        'label/width': common._int64_feature(lbl.shape[1]),
        'label/channels': common._int64_feature(lbl.shape[2])
        }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

class ImgImgParser():
    def __init__(self, feature_folder, label_folder, split=[0.8, 0.1, 0.1]):
        self._feature_folder = feature_folder
        self._label_folder = label_folder
        self._split = split

    def _get_image_paths(self, path):
        return fileops.get_all_files(path)

    def create_records(self, output_path, shards=1):
        """Create n_shards tfrecords at output_path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        feature_path = self._get_image_paths(self._feature_folder)
        label_path = self._get_image_paths(self._label_folder)

        # Get the data split
        split = fileops.split_lists([feature_path, label_path], *self._split)
        print(len(split))
        print(len(split[0]))
        print(len(split[0][0]))
        split_names = ['train', 'val', 'test']

        # Try to split up the "shards" between the dataset splits
        test_shards = ceil(shards * self._split[2])
        val_shards = ceil(shards * self._split[1])
        train_shards = ceil(shards * self._split[0])
        n_shards = [train_shards, val_shards, test_shards]

        # Wrap shard function for readability
        def shard(the_list):
            return listops.chunks(the_list, shards)

        # Create the tfrecords names

        # Loop through
        for _name, (feature_paths, label_paths), _n_shard in zip(split_names, split, n_shards):
            # Create the tfrecord name for each shard
            tf_names = [os.path.join(output_path, '%s_shard_%d.tfrecords' % (_name, i)) for i in range(_n_shard)]
            sharded_features = listops.chunks(feature_paths, _n_shard)
            sharded_labels = listops.chunks(label_paths, _n_shard)
            common.convert_img_img(tf_names, sharded_features, sharded_labels, _get_img_img_example)


