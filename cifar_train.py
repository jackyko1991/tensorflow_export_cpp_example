from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Jacky Ko'

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""


from datetime import datetime
import time

import tensorflow as tf

import cifar

# global definitions
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './tmp/cifar_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")

# tf.app.flags.DEFINE_string('model_dir', './tmp/my-model',
#                            """Directory where to write model proto """
#                            """ to import in c++""")






# tf.app.flags.DEFINE_string('eval_dir', './tmp/log_eval',
#                            """Directory where to write event logs.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/ckpt',
#                            """Directory where to read model checkpoints.""")

# # Parameters
# display_step = 1
# IMAGE_PIXELS = 32 * 32 * 3

# checkpoint_state_name = "checkpoint_state"
# input_graph_name = "input_graph.pb"
# output_graph_name = "output_graph.pb"


# def placeholder_inputs(batch_size):
#   """Generate placeholder variables to represent the the input tensors.
#   These placeholders are used as inputs by the rest of the model building
#   code and will be fed from the downloaded ckpt in the .run() loop, below.
#   Args:
#     batch_size: The batch size will be baked into both placeholders.
#   Returns:
#     images_placeholder: Images placeholder.
#     labels_placeholder: Labels placeholder.
#   """
#   # Note that the shapes of the placeholders match the shapes of the full
#   # image and label tensors, except the first dimension is now batch_size
#   # rather than the full size of the train or test ckpt sets.
#   # batch_size = -1
#   images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
#                                                          IMAGE_PIXELS))
#   labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

#   return images_placeholder, labels_placeholder


# def calculate_accuracy(logit, labels, feed):
#   # Calculate accuracy
#   acc, n_correct = my_cifar.evaluation(logit, labels)

#   return acc, n_correct


def train():
   """Train CIFAR-10 for a number of steps."""
   with tf.Graph().as_default():
    #  global_step = tf.Variable(0, trainable=False)
    global_step = tf.train.get_or_create_global_step()

    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    # GEt images and labels for CIFAR-10
    # # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar_input.distorted_inputs()

    print('images shape: ', images.get_shape())
    print('labels shape: ', labels.get_shape())

    # Build a graph that computes logits predictions from the inference model
    logits = cifar.inference(images_placeholder)
    print('logits shape: ', logits.get_shape())

    # Calculate loss.
    loss = cifar.loss(logits, labels_placeholder)


    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar.train(loss, global_step)

    # Calculate accuracy
    acc, n_correct = cifar.evaluation(logits, labels_placeholder)

    # Create a saver
    saver = tf.train.Saver()

    tf.summary.scalar('Acc', acc)
    tf.summary.scalar('Loss', loss)
    tf.summary.image('Images', tf.reshape(images, shape=[-1, 32, 32, 3]), max_outputs=10)
    tf.summary.image('Val Images', tf.reshape(val_images, shape=[-1, 32, 32, 3]), max_outputs=10)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dirr,
                                            graph_def=sess.graph_def)

    try:
      while not coord.should_stop():
        print("max steps: " + FLAGS.max_steps)
        for step in xrange(FLAGS.max_steps):
          print("step" +  step)

          images_r, labels_r = sess.run([images, labels])
          images_val_r, labels_val_r = sess.run([val_images, val_labels])

          train_feed = {images_placeholder: images_r,
                        labels_placeholder: labels_r}

          val_feed = {images_placeholder: images_val_r,
                      labels_placeholder: labels_val_r}

          # timer for benchmarking
          start_time = time.time()

          _, loss_value = sess.run([train_op, loss], feed_dict=train_feed)
          duration = time.time() - start_time

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          if step % FLAGS.log_frequency == 0:
            images_per_sec = FLAGS.batch_size / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.6f (%.1f images/sec; %.3f sec/batch)')
            print_str_loss = format_str % (datetime.now(), step, loss_value, images_per_sec, sec_per_batch)
            print (print_str_loss)
            summary_str = sess.run([summary_op], feed_dict=train_feed)
            summary_writer.add_summary(summary_str[0], step)

#             if step % (display_step * 5) == 0:
#               acc_value, num_corroect = sess.run([acc, n_correct], feed_dict=train_feed)

#               format_str = '%s: step %d,  train acc = %.2f, n_correct= %d'
#               print_str_train = format_str % (datetime.now(), step, acc_value, num_corroect)
#               print (print_str_train)

#             # Save the model checkpoint periodically.
#             if (step + 1) % (display_step * 10) == 0 or (step + 1) == FLAGS.max_steps:
#               val_acc_r, val_n_correct_r = sess.run([acc, n_correct], feed_dict=val_feed)  # , feed_dict=val_feed

#               frmt_str = 'Step %d, Val Acc = %.2f, num correct = %d'
#               print_str_val = frmt_str % (step, val_acc_r, val_n_correct_r)
#               print(print_str_val)

#               # checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
#               # saver.save(sess, checkpoint_path, global_step=step, latest_filename=checkpoint_state_name)

#               checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "saved_checkpoint")
#               print("saving session in step " + step)
#               saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)

#       except tf.errors.OutOfRangeError:
#         print ('Done training -- epoch limit reached')

#       finally:
#         # When done, ask the threads to stop.
#         coord.request_stop()

#         '''
#          TODO #3.1: Start freezing the graph when training finished
#         '''
#         freeze_my_graph(sess)

#       # Wait for threads to finish.
#       coord.join(threads)
#       sess.close()

#     # If you define TRAIN argument to False, so it will load from the checkpoint file and freeze.
#     else:
#       '''
#          TODO #3.2: You can also freeze the graph from the latest checkpoint if you don't want to wait for a long time.
#       '''
#       freeze_my_graph(sess)

# def freeze_my_graph(sess):

#   tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.model_dir, input_graph_name)

#   # We save out the graph to disk, and then call the const conversion
#   # routine.

#   checkpoint_prefix = posixpath.join(FLAGS.checkpoint_dir, "saved_checkpoint")
#   input_graph_path = posixpath.join(FLAGS.model_dir, input_graph_name)
#   input_saver_def_path = ""
#   input_binary = False
#   input_checkpoint_path = checkpoint_prefix + "-0"
#   # input_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt') + "-0"
#   # input_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-299')
#   output_node_names = "Dense2/output_node"
#   restore_op_name = "save/restore_all"
#   filename_tensor_name = "save/Const:0"
#   output_graph_path = posixpath.join(FLAGS.model_dir, output_graph_name)
#   clear_devices = False

#   freeze_graph.freeze_graph(input_graph_path,
#                             input_saver_def_path,
#                             input_binary,
#                             input_checkpoint_path,
#                             output_node_names,
#                             restore_op_name,
#                             filename_tensor_name,
#                             output_graph_path,
#                             clear_devices)


def main(argv=None):
  cifar.download_and_extract()

  # clear the training log directory
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
