from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000
NUM_EPOCHS_PER_DECAY = 5      # Epochs after which learning rate
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

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
    dtype = tf.float32
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
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def weight_variables(shape, stddev=0.1):
  """stddev of initial weights should be juedged later"""
  #return tf.get_variable("weights", initializer=tf.truncated_normal(shape, stddev))
  return _variable_with_weight_decay('weights', shape=shape, stddev=5e-2, wd=0.0)

def bias_variable(shape, stddev=0.1):
  return tf.get_variable("bias", initializer=tf.truncated_normal(shape, stddev))

def conv2d(x, W):
  """padding item should be judged later"""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                        strides=[1, 2, 2, 1], padding='SAME')

def local_layer(x, kernel_size, map_num, p , q, RELU=False):
  """common model for locally shared weights concolutional layer
  Args:
    x -- input
    kernel_size -- size for weights
    p -- locally weight division numbers, dim0
    q -- locally weight division numbers, dim1
    RELU -- weather to use Relu function
  Returns:
    output -- output of locally weighted convolutional layer
   """
  _, h, w, d = x.get_shape(); h = int(h); w = int(w); d = int(d)
  h_step = int((h - kernel_size + 1) / p)
  w_step = int((w - kernel_size + 1) / q)
  h_convll = []; bias = bias_variable([map_num])
  for i in range(p):
    h_convl = []
    for j in range(q):
      #pdb.set_trace()
      with tf.variable_scope('local_region_%d%d' %(i,j)):
        region = x[:, h_step*i:h_step*(i+1) + kernel_size - 1, 
                      w_step*j:w_step*(j+1) + kernel_size - 1, :]
        region = tf.reshape(region, [-1, h_step + kernel_size - 1, w_step + kernel_size - 1, d])
        W_conv = weight_variables([kernel_size, kernel_size, d, map_num])
        h_convl.append(conv2d(tf.to_float(region), W_conv))
    h_convll.append(h_convl)

  for i in range(p):
    h_convll[i] = tf.concat(1, h_convll[i])
  h_convll = tf.concat(2, h_convll)
  output = tf.nn.tanh(h_convll + bias)
  if RELU: output = tf.nn.relu(output)
  return output

def full_connect(x, n, flat=True):
  """general model for fully connected layer
  Args:
    x -- input
    n -- num of map
    flat -- if True, past layer is conv layer
  Returns:
    h_fc -- output of fully connected layer
  """
  if flat:
    _, h, w, d = x.get_shape(); h = int(h); w = int(w); d = int(d)
    dim = h * w * d
  else:  dim = int(x.get_shape()[1])

  W_fc = weight_variables([dim, n]); b_fc = bias_variable([n])
  x_flat = tf.reshape(x, [-1, dim])  
  h_fc = tf.nn.tanh(tf.matmul(x_flat, W_fc) + b_fc)
  return h_fc

def Net_S0(images):
  """ S0 Net model in TangXiaoOu point detect paper
  while the max_pooling here is simplified. So as in S1 and S2.
  """
  x_input = tf.reshape(images, [-1, 39, 39, 3])
  
  with tf.variable_scope('conv1'):
    h_conv1 = local_layer(x_input, 4, 20, 2, 2, True)
  h_pool1 = max_pool_2x2(h_conv1)
  with tf.variable_scope('conv2'):
    h_conv2 = local_layer(h_pool1, 3, 40, 2, 2, True)
  h_pool2 = max_pool_2x2(h_conv2)
  with tf.variable_scope('conv3'):
    h_conv3 = local_layer(h_pool2, 3, 60, 3, 3)
  h_pool3 = max_pool_2x2(h_conv3)
  with tf.variable_scope('conv4'):
    h_conv4 = local_layer(h_pool3, 2, 80, 2, 2)
  with tf.variable_scope('fully_connected1'):
    h_fc1 = full_connect(h_conv4, 120)
  with tf.variable_scope('fully_connected2'):
    h_fc2 = full_connect(h_fc1, 10, False)
  return h_fc2

def Net_S1(images):
  with tf.variable_scope('input'):
    x_input = tf.reshape(images, [-1, 31, 39, 3])
  with tf.variable_scope('conv1'):
    h_conv1 = local_layer(x_input, 4, 20, 1, 1, True)
  with tf.variable_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
  with tf.variable_scope('conv2'):
    h_conv2 = local_layer(h_pool1, 3, 40, 2, 2, True)
  with tf.variable_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
  with tf.variable_scope('conv3'):
    h_conv3 = local_layer(h_pool2, 3, 60, 2, 3)
  with tf.variable_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv2)
  with tf.variable_scope('conv4'):
    h_conv4 = local_layer(h_pool3, 2, 80, 1, 2)
  with tf.variable_scope('fully_connected1'):
    h_fc1 = full_connect(h_conv4, 100)
  with tf.variable_scope('fully_connected2'):
    h_fc2 = full_connect(h_fc1, 6, False)
  #pdb.set_trace()
  return h_fc2

def Net_S2(images):
  with tf.variable_scope('input'):
    x_input = tf.reshape(images, [-1, 15, 15, 3])
  with tf.variable_scope('conv1'):
    h_conv1 = local_layer(x_input, 4, 20, 1, 1, True)
  with tf.variable_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
  with tf.variable_scope('conv2'):
    h_conv2 = local_layer(h_pool1, 3, 40, 1, 1, True)
  with tf.variable_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
  with tf.variable_scope('fully_connected1'):
    h_fc1 = full_connect(h_pool2, 60)
  with tf.variable_scope('fully_connected2'):
    h_fc2 = full_connect(h_fc1, 2, False)

  return h_fc2

def inference(images, batch_size=100):
  """construction of the whole graph.
  Args:
    images -- 39*39 sized images, feed by input_data.py
  Returns:
    final_preds -- preds in 39 * 39 images, will be rescale in running the graph
  """
  zero = tf.constant(0.0, shape=[1, 1]); neg1 = tf.constant(-1.0, shape=[1,1])
  zeros = tf.constant(0.0, shape=[batch_size, 4])
  with tf.variable_scope('Level1'):
    with tf.variable_scope('F1'):
      F1_preds = Net_S0(images)
    with tf.variable_scope('EN1'):
      EN1_inp = tf.slice(images, [0,0,0,0], [-1, 31, 39, -1])
      EN1_preds_raw = Net_S1(EN1_inp)
      #pdb.set_trace()
      EN1_preds = tf.concat(1, [EN1_preds_raw, zeros])
    with tf.variable_scope('NM1'):
      NM1_inp = tf.slice(images, [0,8,0,0], [-1, 39, 39, -1])
      NM1_preds_raw = Net_S1(EN1_inp)
      NM1_preds = tf.concat(1, [zeros, NM1_preds_raw])
  Level1_raw = tf.add(F1_preds, EN1_preds); Level1_raw = tf.mul(tf.add(Level1_raw, NM1_preds), 39)
  Level1_preds = tf.div(Level1_raw, [2,2,2,2,3,3,2,2,2,2])

  def out_of_range(begin, sizes, part, size):
    """Judge whether the crop range related to last layer is out of the range of 39*39 images
    If either coordinate is out, push it back.

    Args:
      begin -- begin tensor, param for tf.slice
      sizes -- size tensor, param for tf.slice
      part -- decide whether the part is left eye or etc. out range of different part is treated differently
            0 ~ 4
      size -- size of crop box(square)

    Returns:
      begin_jdg -- judged tensor for begin param in tf.slice
    """

    left = begin[0]; top = begin[1]; right = tf.add(begin[0], sizes[0]); bottom = tf.add(begin[1], sizes[1])
    if part == 0 or part == 3: 
      left_jdg = tf.maximum(0, left); left_jdg = tf.minimum(39 - size, left_jdg)
      if part == 0: 
        top_jdg = tf.maximum(0, top); top_jdg = tf.minimum(39 - size, top_jdg)
      else: 
        bottom_jdg = tf.minimum(39, bottom); bottom_jdg = tf.maximum(size, bottom_jdg)
        top_jdg = tf.add(bottom_jdg, tf.neg(sizes[1]))
    elif part == 1 or part == 4: 
      right_jdg = tf.minimum(39, right); right_jdg = tf.maximum(right_jdg, size)
      left_jdg = tf.add(right_jdg, tf.neg(sizes[0]))
      if part == 1: top_jdg = tf.maximum(0, top); top_jdg = tf.minimum(top_jdg, 39 - size)
      else: 
        bottom_jdg = tf.minimum(39, bottom); bottom_jdg = tf.maximum(size, bottom_jdg)
        top_jdg = tf.add(bottom_jdg, tf.neg(sizes[1]))
    else: left_jdg = int((39 - size) / 2); top_jdg = left_jdg
    begin_jdg = tf.convert_to_tensor([left_jdg, top_jdg, 0])

    return begin_jdg

  def crop_and_resize(center, size, part):
    """crop and resize the area of interest of diff part.

    Args:
      center -- center of the crop box. output of last level based on the paper.
      size -- size of crop box
      part -- 0 ~ 4, decide left eye or etc

    Returns:
      cropped_images -- cropped and resized area of interest, input of next level
      crds -- crop box coordinates, saved for rescale.
    """

    images_list = tf.split(0, batch_size, images); size_l = int(- size * 0.5)
    offsets_raw = tf.to_float(tf.constant(size_l, shape=[batch_size, 2]))
    crds = tf.add(center, offsets_raw); offsets_list = tf.split(0, batch_size, crds)
    cropped_images = []
    for i in range(batch_size):
      offset = tf.squeeze(offsets_list[i])
      image = tf.squeeze(images_list[i]); begin_raw = tf.to_int32(tf.convert_to_tensor([offset[0], offset[1], 0]))
      sizes = tf.convert_to_tensor([size, size, -1])
      begin = out_of_range(begin_raw, sizes, part, size)
      cropped_image = tf.image.resize_images(tf.slice(image, begin, sizes), 15, 15)
      cropped_images.append(cropped_image)

    cropped_images = tf.concat(0, cropped_images)
    return cropped_images, crds
    
  Level2_preds = []
  with tf.variable_scope('Level2'):
    factors1 = tf.constant(15 / 10, shape=[batch_size, 2])
    factors2 = tf.constant(1.25, shape=[batch_size, 2])
    for i in range(5):
      with tf.variable_scope('Level2_%d' %i):
        with tf.variable_scope('pred1'):
          inp1, crd1 = crop_and_resize(Level1_preds[:,2*i:2*i+2], 10, i)

          preds1_raw = tf.mul(15.0, Net_S2(inp1))
          preds1 = tf.div(preds1_raw, factors1)
          preds1 = tf.add(preds1, crd1)
        with tf.variable_scope('preds2'):
          inp2, crd2 = crop_and_resize(Level1_preds[:,2*i:2*i+2], 12, i)
          preds2_raw = tf.mul(15.0, Net_S2(inp2))
          preds2 = tf.div(preds2_raw, factors2)
          preds2 = tf.add(preds2, crd2)
      Level2_preds.append(tf.reduce_mean([preds1, preds2], 0))
  
  Level3_preds = []
  with tf.variable_scope('Level3'):
    factors1 = tf.constant(15.0 / 7.0, shape=[batch_size, 2])
    factors2 = tf.constant(15.0 / 8.0, shape=[batch_size, 2])
    for i in range(5):
      with tf.variable_scope('Level3_%d' %i):
        with tf.variable_scope('pred1'):
          inp1, crd1 = crop_and_resize(Level2_preds[i], 7, i)
          preds1_raw = tf.mul(15.0, Net_S2(inp1))
          preds1 = tf.div(preds1_raw, factors1)
          preds1 = tf.add(preds1, crd1)
        with tf.variable_scope('preds2'):
          inp2, crd2 = crop_and_resize(Level2_preds[i], 8, i)
          preds2_raw = tf.mul(15.0, Net_S2(inp2))
          preds2 = tf.div(preds2_raw, factors2)
          preds2 = tf.add(preds2, crd2)
      Level3_preds.append(tf.reduce_mean([preds1, preds2], 0))

  Level2_preds = tf.concat(1, Level2_preds);  Level3_preds = tf.concat(1, Level3_preds)
  final_preds = tf.add(Level1_preds, Level2_preds)
  final_preds = tf.div(tf.add(final_preds, Level3_preds), 3.0)

  return final_preds
  
def training(loss, global_step):
  """Sets up the training Ops.
  """
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def train(total_loss, global_step):
  """Train model with decay learning rate and batch gradient descent.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / 100
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.

  # Compute gradients.
  with tf.control_dependencies([total_loss]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  return train_op

def loss(bbox_widths, preds, points, batch_size=100):
  """loss function based on paper, returns a tensor of batch_size.
  """
  diff = tf.squared_difference(preds, points)
  dist = []
  for i in range(5):
    dist.append(tf.reshape(tf.reduce_sum(diff[:,2*i:2*i+2], 1), [batch_size, 1]))
  dist = tf.reduce_sum(tf.sqrt(tf.concat(1, dist)), 1)
  error = tf.div(dist, bbox_widths)
  return error

def evaluation(loss, batch_size=100):
  """Evaluate the quality of the logits at predicting the label.
  """
  ruler = tf.constant(0.5, shape=[batch_size, 1])
  loss_l = tf.reshape(loss, [batch_size, 1])
  comp = tf.concat(1, [ruler, loss_l])
  correct = tf.argmin(comp, 1)
  # Return the number of entries less than 0.5
  return tf.reduce_sum(correct)
