import tensorflow as tf
import src.tfrecord.common as common
from src.lib import fileops, imgops, listops
import os

def _get_img_img_example(self, feature_path, label_path):
    """Create the feature/label pair tf example"""

    ftr = imgops.load_image(i_ftr)
    lbl = imgops.load_image(i_lbl)
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
    
    return

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

        def shard(the_list):
            return listops.chunks(the_list, shards)

        # Loop through
        for _type, (ftr, lbl) in zip(['train', 'val', 'test'], split):
            for i, (shard_ftr, shard_lbl) in enumerate(zip(shard(ftr), shard(lbl))):
                tf_record_name = os.path.join(output_path, '%s_shard_%d.tfrecords' % (_type, i))
                common.convert_img_img(tf_record_name, shard_ftr, shard_lbl, _get_img_img_example)


