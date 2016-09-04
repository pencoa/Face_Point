from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import string
from tensorflow.python.framework import dtypes
import Image

ranges = [-0.05, 1.05, -0.05, 1.05]

def extract_data(path):
  """ extract all data for feeding placeholder.
  Args:
    path -- the txt file each line with format
            image_dir bounding box labeled points
  Return:
    images -- image cropped around face range and resized to 39*39
    points -- points extracted from txt 
    factors -- scale factor
    crds -- crop box
    widths -- bounding box width 
  """
  images = []; points = []; factors = []; crds = []; widths = []
  with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
      a = line.split(); impath = a[0]; a = a[1:]
      aa = []
      for i in range(len(a)): aa.append(string.atof(a[i]))
      bbox_width = aa[1] - aa[0]; bbox_height = aa[3] - aa[2]
      widths.append(bbox_width)
      left = int(bbox_width * ranges[0] + aa[0])
      top = int(bbox_height * ranges[2] + aa[2])
      if bbox_height >= bbox_width: 
        bottom = int(bbox_height * ranges[3] + aa[2])
        height = bottom - top; diff = bbox_height - bbox_width
        left = int(left - 0.5 * diff); right = left + height
        factor = 39 / height
      else: 
        right = int(bbox_width * ranges[1] + aa[0])
        width = right - left; diff = bbox_width - bbox_height
        top = int(top - 0.5*diff); bottom = top + width
        factor = 39 / width
      factors.append([factor])
      box = (left, right, top, bottom); crds.append([left, top])
      im = Image.open(impath); image = im.crop(box)
      images.append(np.array(image.resize((39, 39))) / 255)
      point_raw = aa[4:];      points.append(point_raw)
  print(points[0])
  return images, points, factors, crds, widths

class DataSet(object):

  def __init__(self,
               images,
               points,
               factors,
               crds,
               widths,
               dtype=dtypes.float32):
    """Construct a DataSet."""

    self._num_examples = len(images)
    self._images = images
    self._points = points
    self._factors = factors
    self._widths = widths
    self._crds = crds
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._current_batch = []

  @property
  def images(self):
    return self._images

  @property
  def points(self):
    return self._points

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, test=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      if not test:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        def rearange(lists, permu):
          new_list = []
          for i in permu: new_list.append(lists[i])
          return new_list

        self._images = rearange(self._images, perm)
        self._factors = rearange(self._factors, perm)
        self._crds = rearange(self._crds, perm)
        self._widths = rearange(self._widths, perm)
        self._points = rearange(self._points, perm)
        # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._points[start:end], self._factors[start:end], self._crds[start:end], self._widths[start:end]

def read_data_sets(dtype=dtypes.float32):
  TRAIN_IMAGES = 'trainImageList.txt'
  VALIDATION_IMAGES = 'testImageList.txt'

  train_images, train_points, train_factors, train_crds, train_widths = extract_data(TRAIN_IMAGES)
  validation_images, validation_points, validation_factors, validation_crds, validation_widths = extract_data(VALIDATION_IMAGES)
  
  train = DataSet(train_images, train_points, train_factors, train_crds, train_widths, dtype=dtype)
  validation = DataSet(validation_images, validation_points, validation_factors, 
                                validation_crds, validation_widths, dtype=dtype)
  
  return train, validation

def read_test(dtype=dtypes.float32):
  TEST_IMAGES = 'lfpw_test_249_bbox.txt'
  test_images, test_points, test_factors, test_crds, test_widths = extract_data(TEST_IMAGES)
  num = len(test_factors)
  test_points = [num * [0,0,0,0,0,0,0,0,0,0]]
  test_points = np.reshape(test_points, [num, 10])
  test = DataSet(test_images, test_points, test_factors, test_crds, test_widths, dtype=dtype)
  return test, num