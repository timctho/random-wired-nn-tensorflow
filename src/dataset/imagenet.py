# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

import glob
import os
import tensorflow as tf  # pylint: disable=g-bad-import-order

from src.dataset import imagenet_preprocessing

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return glob.glob(os.path.join(data_dir, 'train/*'))
        # return [
        #     os.path.join(data_dir, 'train-%05d-of-01024' % i)
        #     for i in range(_NUM_TRAIN_FILES)]
    else:
        return glob.glob(os.path.join(data_dir, 'validation/*'))
        # return [
        #     os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        #     for i in range(128)]


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
    }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(serialized=example_serialized,
                                          features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.
      dtype: data type to use for images/features.

    Returns:
      Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=DEFAULT_IMAGE_SIZE,
        output_width=DEFAULT_IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        is_training=is_training)
    image = tf.cast(image, dtype)

    return image, label


def input_fn(is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             num_parallel_batches=1,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False):
    """Input function which provides batches for train or eval.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features
      datasets_num_private_threads: Number of private threads for tf.data.
      num_parallel_batches: Number of parallel batches for tf.data.
      parse_record_fn: Function to use for parsing the records.
      input_context: A `tf.distribute.InputContext` object passed in by
        `tf.distribute.Strategy`.
      drop_remainder: A boolean indicates whether to drop the remainder of the
        batches. If True, the batch dimension will be static.

    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if input_context:
        tf.logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
                input_context.input_pipeline_id, input_context.num_input_pipelines))
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records.
    # cycle_length = 10 means that up to 10 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=4)

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record_fn,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=num_parallel_batches,
        drop_remainder=drop_remainder
    )


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.

    Args:
      resnet_size: The number of convolutional layers needed in the model.

    Returns:
      A list of block sizes to use in building the model.

    Raises:
      KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(resnet_size, choices.keys()))
        raise ValueError(err)


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           num_parallel_batches=1,
                           drop_remainder=False):
    """Given a Dataset with raw records, return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features.
      datasets_num_private_threads: Number of threads for a private
        threadpool created for all datasets computation.
      num_parallel_batches: Number of parallel batches for tf.data.
      drop_remainder: A boolean indicates whether to drop the remainder of the
        batches. If True, the batch dimension will be static.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """
    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = (
            datasets_num_private_threads)
        dataset = dataset.with_options(options)
        tf.logging.info('datasets_num_private_threads: %s',
                        datasets_num_private_threads)

    # Disable intra-op parallelism to optimize for throughput instead of latency.
    # options = tf.data.Options()
    # options.experimental_threading.max_intra_op_parallelism = 1
    # dataset = dataset.with_options(options)

    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    # Parses the raw records into images and labels.
    dataset = dataset.map(
        lambda value: parse_record_fn(value, is_training, dtype),
        num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=1000)

    return dataset


def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32):
    """Returns an input function that returns a dataset with random data.

    This input_fn returns a data set that iterates over a set of random data and
    bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
    copy is still included. This used to find the upper throughput bound when
    tunning the full input pipeline.

    Args:
      height: Integer height that will be used to create a fake image tensor.
      width: Integer width that will be used to create a fake image tensor.
      num_channels: Integer depth that will be used to create a fake image tensor.
      num_classes: Number of classes that should be represented in the fake labels
        tensor
      dtype: Data type for features/images.

    Returns:
      An input_fn that can be used in place of a real one to return a dataset
      that can be used for iteration.
    """

    # pylint: disable=unused-argument
    def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
        """Returns dataset filled with random data."""
        # Synthetic input should be within [0, 255].
        inputs = tf.random.truncated_normal(
            [batch_size] + [height, width, num_channels],
            dtype=dtype,
            mean=127,
            stddev=60,
            name='synthetic_inputs')

        labels = tf.random.uniform(
            [batch_size],
            minval=0,
            maxval=num_classes - 1,
            dtype=tf.int32,
            name='synthetic_labels')
        data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
        data = data.prefetch(1000)
        return data

    return input_fn


if __name__ == '__main__':
    import cv2
    import numpy as np

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

    dataset = input_fn(True, '/Tim/Data/ImageNet', 8)
    images, labels = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        img_np, labels_np = sess.run([images, labels])

        print(img_np.shape, labels_np.shape)

        for i in range(img_np.shape[0]):
            cur_img = img_np[i]
            cur_label = labels_np[i]
            print(cur_label)

            cur_img = cur_img + np.array(_CHANNEL_MEANS)

            print(np.max(cur_img), np.min(cur_img))

            cv2.imshow('', cur_img[:,:,::-1].astype(np.uint8))
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

