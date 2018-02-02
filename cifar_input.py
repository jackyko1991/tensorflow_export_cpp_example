from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Jacky Ko'

"""Routine for decoding the CIFAR-10 binary file format."""

import tensorflow as tf
import os
import glob
import numpy as np
import csv
from six.moves import xrange
import multiprocessing

# Process images at the defined size. Note that original cifar 10 data image size is 32*32.
# If you alter this number, the entire model architecture will change and the model would need to retrain
IMAGE_SIZE = 32

# Global constants descrbing the CIFAR-10 dataset
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


# IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, 3]
# # Basic model parameters as external flags.
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')
# flags.DEFINE_integer('batch_size', 128, 'Batch size.')
# flags.DEFINE_string('train_dir', '../my_data_raw', 'Directory with the training ckpt.')

# # Constants used for dealing with the files, matches convert_to_records.
# TRAIN_FILE = 'train.tfrecords'
# VALIDATION_FILE = 'validation.tfrecords'


# def _dense_to_one_hot(labels_dense, num_classes):
#   """Convert class labels from scalars to one-hot vectors."""
#   num_labels = labels_dense.shape[0]
#   index_offset = np.arange(num_labels) * num_classes
#   labels_one_hot = np.zeros((num_labels, num_classes))
#   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#   print(labels_one_hot[0])
#   return labels_one_hot


# def _label_to_int(labels):
#   categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#   new_labels = []

#   for label in labels:
#     new_labels.append(categories.index(label[1]))
#   return new_labels


# '''Read Images and Labels normally, with python :))) :D'''


# def read_labels_from(path=Data_PATH, num_classes=num_classes, one_hot=False):
#   print('Reading labels')
#   with open(os.path.join(path, 'trainLabels.csv'), 'r') as dest_f:
#     data_iter = csv.reader(dest_f)
#     train_labels = [data for data in data_iter]

#   # pre process labels to int
#   train_labels = _label_to_int(train_labels)
#   train_labels = np.array(train_labels, dtype=np.uint32)

#   # if one_hot:
#   #   labels_one_hot = _dense_to_one_hot(train_labels, num_classes)
#   #   labels_one_hot = np.asarray(labels_one_hot)
#   #   return labels_one_hot

#   return train_labels


# def read_images_from(path=Data_PATH):
#   images = []
#   png_files_path = glob.glob(os.path.join(path, 'train/', '*.[pP][nN][gG]'))
#   for filename in png_files_path:
#     im = Image.open(filename)  # .convert("L")  # Convert to greyscale
#     im = np.asarray(im, np.uint8)
#     # print(type(im))
#     # get only images name, not path
#     image_name = filename.split('/')[-1].split('.')[0]
#     images.append([int(image_name), im])

#   images = sorted(images, key=lambda image: image[0])

#   images_only = [np.asarray(image[1], np.uint8) for image in images]  # Use unint8 or you will be !!!
#   images_only = np.array(images_only)

#   print(images_only.shape)
#   return images_only


# def inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs):
#   """Reads input ckpt num_epochs times.
#   Args:
#     train: Selects between the training (True) and validation (False) ckpt.
#     batch_size: Number of examples per returned batch.
#     num_epochs: Number of times to read the input ckpt, or 0/None to
#        train forever.
#   Returns:
#     A tuple (images, labels), where:
#     * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
#       in the range [-0.5, 0.5].
#     * labels is an int32 tensor with shape [batch_size] with the true label,
#       a number in the range [0, mnist.NUM_CLASSES).
#     Note that an tf.train.QueueRunner is added to the graph, which
#     must be run using e.g. tf.train.start_queue_runners().
#   """
#   if not num_epochs: num_epochs = None
#   filename = os.path.join(FLAGS.train_dir,
#                           TRAIN_FILE if train else VALIDATION_FILE)

#   with tf.name_scope('input'):
#     filename_queue = tf.train.string_input_producer(
#       [filename], num_epochs=num_epochs, name='string_input_producer')

#     # Even when reading in multiple threads, share the filename
#     # queue.
#     image, label = read_and_decode(filename_queue)
#     print('1- image shape is ', image.get_shape())

#     # Ensure that the random shuffling has good mixing properties.
#     min_fraction_of_examples_in_queue = 0.4
#     min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
#                              min_fraction_of_examples_in_queue)

#     images, sparse_labels = tf.train.shuffle_batch(
#       [image, label], batch_size=batch_size, num_threads=5,
#       capacity=min_queue_examples + 3 * batch_size, enqueue_many=False,
#       # Ensures a minimum amount of shuffling of examples.
#       min_after_dequeue=min_queue_examples, name='batching_shuffling')
#     print('1.1- label batch shape is ', sparse_labels.get_shape())

#     return images, sparse_labels


# def inputs_val(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs):
#   """Reads input ckpt num_epochs times.
#   Args:
#     train: Selects between the training (True) and validation (False) ckpt.
#     batch_size: Number of examples per returned batch.
#     num_epochs: Number of times to read the input ckpt, or 0/None to
#        train forever.
#   Returns:
#     A tuple (images, labels), where:
#     * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
#       in the range [-0.5, 0.5].
#     * labels is an int32 tensor with shape [batch_size] with the true label,
#       a number in the range [0, mnist.NUM_CLASSES).
#     Note that an tf.train.QueueRunner is added to the graph, which
#     must be run using e.g. tf.train.start_queue_runners().
#   """
#   if not num_epochs:
#     num_epochs = None

#   filename = os.path.join(FLAGS.train_dir,
#                           VALIDATION_FILE)

#   with tf.name_scope('input'):
#     filename_queue = tf.train.string_input_producer(
#       [filename])

#     # Even when reading in multiple threads, share the filename
#     # queue.
#     image, label = read_and_decode(filename_queue)
#     print('1- image shape is ', image.get_shape())

#     # Shuffle the examples and collect them into batch_size batches.
#     # (Internally uses a RandomShuffleQueue.)
#     # We run this in two threads to avoid being a bottleneck.
#     images, sparse_labels = tf.train.shuffle_batch(
#       [image, label], batch_size=batch_size, num_threads=3,
#       capacity=300 + 3 * batch_size, enqueue_many=False,
#       # Ensures a minimum amount of shuffling of examples.
#       min_after_dequeue=300)
#     print('1.1- images batch shape is ', images.get_shape())

#     return images, sparse_labels

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = multiprocessing.cpu_count()

  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def read_cifar10(filename_queue):
  """
  Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.

  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in xrange(1, 6)]

  # check all files are downloaded and unziped properly
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with tf.name_scope('data_augumentation'):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    #   # Image processing for training the network. Note the many random
    # distortions applied to the image.
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(resized_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                              lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Reshape to [32, 32, 3] as distortion methods need this shape
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print ('Filling queue with %d CIFAR images before starting to train. '
           'This may take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                        min_queue_examples, batch_size,
                                        shuffle=True)

def main(argv=None):
  return 0

if __name__ == '__main__':
  tf.app.run()
