from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Jacky Ko'
"""
Build teh CIFAR-10 network 

Summary of available functions:

# Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)

"""

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar_input.IMAGE_SIZE
NUM_CLASSES = cifar_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# # Network Parameters
# n_input = 32 * 32 * 3  # Cifar ckpt input (img shape: 32*32)

# out_conv_1 = 64
# out_conv_2 = 64

# n_hidden_1 = 384
# n_hidden_2 = 192

# dropout = 0.90  # Dropout, probability to keep units

# # Noise level
# noise_level = 20

# # Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 10.0  # Epochs after which learning rate decays.
# LEARNING_RATE_DECAY_FACTOR = 0.60  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.001  # 954 Initial learning rate.


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def inference(images):
  """Build the CIFAR model up to where it may be used for inference.
  Args:
    images: Images returned from distorted_inputs() or inputs()

  Returns:
    logits: Output tensor with the computed logits.
  """

  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                        shape = [5, 5, 3, 64],
                                        stddev=5e-2,
                                        wd=None)

    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

   # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy across the batch
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def training(loss, global_step):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
#   # Variables that affect learning rate.
#   num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
#   decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

#   print('Decay steps is: ', decay_steps)
#   # Decay the learning rate exponentially based on the number of steps.
#   lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                   global_step,
#                                   decay_steps,
#                                   LEARNING_RATE_DECAY_FACTOR,
#                                   staircase=True)
#   tf.summary.scalar('learning_rate', lr)
#   # Add a scalar summary for the snapshot loss.
#   tf.summary.scalar(loss.op.name, loss)
#   # Create the gradient descent optimizer with the given learning rate.

#   optimizer = tf.train.AdamOptimizer(lr)
#   # optimizer = tf.train.GradientDescentOptimizer(lr)

#   # Use the optimizer to apply the gradients that minimize the loss
#   # (and also increment the global step counter) as a single training step.
#   train_op = optimizer.minimize(loss, global_step=global_step)

#   return train_op


# def evaluation(logits, labels):
#   """Evaluate the quality of the logits at predicting the label.
#   Args:
#     logits: Logits tensor, float - [batch_size, NUM_CLASSES].
#     labels: Labels tensor, int32 - [batch_size], with values in the
#       range [0, NUM_CLASSES).
#   Returns:
#     A scalar int32 tensor with the number of examples (out of batch_size)
#     that were predicted correctly.
#   """
#   print('Evaluation..')
#   # For a classifier model, we can use the in_top_k Op.
#   # It returns a bool tensor with shape [batch_size] that is true for
#   # the examples where the label's is was in the top k (here k=1)
#   # of all logits for that example.
#   correct = tf.nn.in_top_k(logits, labels, 1)
#   num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
#   # print(num_correct)
#   acc_percent = num_correct / FLAGS.batch_size

#   # Return the number of true entries.
#   return acc_percent * 100.0, num_correct  # Changed reduce_sum() to reduce_mean()

def distorted_inputs():
  """
  Construct distorted input for CIFAR training using the Reader ops.
  Raises:
    ValueError: if no data_dir

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not FLAGS.data_dir:
    raise ValueError('Please provide the data_dir')

  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar_input.distorted_inputs(data_dir=data_dir,
                                            batch_size=FLAGS.batch_size)
  
  if FLAGS.use_fp16:
    images = tf.cast(images,tf.float16)
    labels = tf.cast(images,tf.float16)
  
  return images, labels

def download_and_extract():
  """
  Download and extract tarball from cifar website
  """
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.mkdir(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory,filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      """helper function to display the download progress"""
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    # unzip the tar file
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin').replace("\\","/")
    if not os.path.exists(extracted_dir_path):
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def main(argv=None):
  return 0

if __name__ == '__main__':
  tf.app.run()
