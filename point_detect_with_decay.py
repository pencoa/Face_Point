from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import input_data
import face_point_with_decay
import time
import numpy
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors."""
  images_placeholder = tf.placeholder(tf.int32, shape=[batch_size, 39, 39, 3])
  points_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 10])
  factors_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 1])
  crds_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 2])
  width_placeholder = tf.placeholder(tf.float32, shape=[batch_size])

  return images_placeholder, points_placeholder, factors_placeholder, crds_placeholder, width_placeholder

def fill_feed_dict(data_set, images_pl, points_pl, factors_pl, crds_pl, width_pl, test=False):
  """Fills the feed_dict for training the given step."""
  images_feed, points_feed, factors_feed, crds_feed, width_feed = data_set.next_batch(
                                    FLAGS.batch_size, test)
  feed_dict = {
      images_pl: images_feed,
      points_pl: points_feed,
      factors_pl: factors_feed,
      crds_pl: crds_feed,
      width_pl: width_feed
  }   
  return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            points_placeholder,
            factors_placeholder, 
            crds_placeholder,
            width_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               points_placeholder,
                               factors_placeholder, 
                               crds_placeholder,
                               width_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  return precision

def run_training():
  """ training and restore model. cross validtion and test is optional at the end of the function."""
  trains, validations = input_data.read_data_sets()

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    images_placeholder, points_placeholder, factors_placeholder, crds_placeholder, width_placeholder = placeholder_inputs(FLAGS.batch_size)

    preds_raw = face_point_with_decay.inference(images_placeholder)
    # change size of factors and crds to fit rescale operation
    factors = tf.concat(1, [factors_placeholder] * 10)
    crdss = tf.concat(1, [crds_placeholder[:,:2]]*5)
    preds = tf.add(crdss, tf.div(preds_raw, factors_placeholder))

    sess = tf.Session()
    # loss vector -- [batch_size]
    loss_s = face_point_with_decay.loss(width_placeholder, preds, points_placeholder)
    losses = tf.reduce_mean(loss_s)

    train_op = face_point_with_decay.train(losses, global_step)
    eval_correct = face_point_with_decay.evaluation(loss_s)
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    sess.run(init)
    step = 0
    while(step < FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and factors
      # for this particular training step.
      feed_dict = fill_feed_dict(trains,
                                 images_placeholder,
                                 points_placeholder,
                                 factors_placeholder, 
                                 crds_placeholder,
                                 width_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      #predict = sess.run(preds, feed_dict=feed_dict)
      #print(predict)

      _, loss_value = sess.run([train_op, losses], feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        prds = sess.run(preds, feed_dict=feed_dict)
        print(prds[0])
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 2000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        pres = do_eval(sess,
                eval_correct,
                images_placeholder,
                points_placeholder,
                factors_placeholder, 
                crds_placeholder,
                width_placeholder,
                trains)
      step += 1
      # Evaluate against the validation set."""
      """print('Validation Data Eval:')
      do_eval(sess,
              eval_correct,
              images_placeholder,
              labels_placeholder,
              factors_placeholder, 
              crds_placeholder,
              validations)
      # Evaluate against the test set.
      print('Test Data Eval:')
      do_eval(sess,
              eval_correct,
              images_placeholder,
              labels_placeholder,
              factors_placeholder, 
              crds_placeholder,
              tests)"""

def run_test():
  """ a simple function to restore model and test, points output to predict.txt."""
  tests, num = input_data.read_test()
  with open('predict.txt', 'w+') as f:
    with tf.Graph().as_default():
      images_placeholder, points_placeholder, factors_placeholder, crds_placeholder, width_placeholder = placeholder_inputs(FLAGS.batch_size)
      preds_raw = face_point_with_decay.inference(images_placeholder)

      factors = tf.concat(1, [factors_placeholder] * 10)
      crdss = tf.concat(1, [crds_placeholder[:,:2]]*5)
      preds = tf.add(crdss, tf.div(preds_raw, factors_placeholder))
      sess = tf.Session()
      saver = tf.train.Saver()
      saver.restore(sess, "data/checkpoint-2999")
      print("Model restored.")
      step = 0
      while(step < num / FLAGS.batch_size):
        feed_dict = fill_feed_dict(tests,
                                   images_placeholder,
                                   points_placeholder,
                                   factors_placeholder, 
                                   crds_placeholder,
                                   width_placeholder,
                                   test=True)
        prds = sess.run(preds, feed_dict=feed_dict)
        for i in range(FLAGS.batch_size):
          prd = prds[i]
          for p in prd: f.write('%.2f ' %p)
          f.write('\n')
        step += 1

def main(_):
  run_training()
  #run_test()

if __name__ == '__main__':
  tf.app.run()
