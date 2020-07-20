import os
import random
import pandas
import numpy
import cv2
import matplotlib.image as mpimg

import sklearn
if sklearn.__version__ < '0.20':
  from sklearn.cross_validation import train_test_split
else:
  from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

SIDE_CAMERA_ANGLE_SHIFT = 0.27
STRAIGHT_ANGLE_THRESHOLD = 0.15
DATA_PATH = './data'

def prepare_data(validation_portion=0.2, big_angles_count=500):
  """
  params[in] validation_portion: the ratio of validation set for data split
  params[in] unbalanced_index: the ideal ratio of sample sizes between 
                               going_straight and turning left/right
  """
  # load data
  center_images, left_images, right_images, angles = _load_data()
  # shuffle and split data
  center_images_, angles_ = center_images[:], angles[:]
  center_images_, angles_ = shuffle(center_images_, angles_)
  X_train, X_valid, y_train, y_valid = train_test_split(
    center_images_, angles_, test_size=validation_portion)  
  
  # calculate sample size for going straight and 
  # going non-straight in training set
  non_straight_count = straight_count = 0
  options = []
  for ind, angle in enumerate(angles):
    if abs(angle) > STRAIGHT_ANGLE_THRESHOLD:
      non_straight_count += 1
      options.append(ind)
    else:
      straight_count += 1
  print("non-straight: {}, stright: {}".format(non_straight_count, straight_count))
   
  # add big angles for recovery
  for _ in range(big_angles_count):
    ind = random.choice(options)
    if angles[ind] < -STRAIGHT_ANGLE_THRESHOLD:
      X_train.append(right_images[ind])
      y_train.append(angles[ind] - SIDE_CAMERA_ANGLE_SHIFT)
    else:
      X_train.append(left_images[ind])
      y_train.append(angles[ind] + SIDE_CAMERA_ANGLE_SHIFT)
  
  X_train, y_train = shuffle(X_train, y_train)
  return X_train, X_valid, y_train, y_valid

def generate_training_data(batch_size, X_train, y_train):
  batch_X = numpy.zeros((batch_size, 64, 64, 3), dtype=numpy.float32)
  batch_y = numpy.zeros((batch_size,), dtype=numpy.float32)
  while True:
    X_train, y_train = shuffle(X_train, y_train)
    for i in range(batch_size):
      ind = random.randrange(len(X_train))
      image = mpimg.imread(os.path.join(DATA_PATH, X_train[ind]))
      # crop the image and then resize it to 64x64
      image_cropped = cv2.resize(image[60: 140,:], (64,64))
      angle = y_train[ind]
      # inject noise into the selected images
      batch_X[i], batch_y[i] = _inject_noise(image_cropped, angle)
    yield batch_X, batch_y

def generate_validation_data(batch_size, X_valid, y_valid):
  batch_X = numpy.zeros((batch_size, 64, 64, 3), dtype=numpy.float32)
  batch_y = numpy.zeros((batch_size,), dtype=numpy.float32)
  while True:
    X_valid, y_valid = shuffle(X_valid, y_valid)
    for i in range(batch_size):
      ind = random.randrange(len(X_valid))
      image = mpimg.imread(os.path.join(DATA_PATH, X_valid[ind].strip()))
      # crop the image and then resize it to 64x64
      batch_X[i] = cv2.resize(image[60: 140,:], (64,64))
      batch_y[i] = y_valid[ind]
    yield batch_X, batch_y


################################################################################
# private functions used only in this file
################################################################################

def _inject_noise(image, angle):
  # apply a random change of the lighting
  image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  image_hsv[:,:,2] = random.uniform(0.3, 1.0) * image_hsv[:,:,2]
  image_light_changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
  # apply a random flip
  if random.randrange(2) == 0:
    image_flipped = cv2.flip(image_light_changed, 1)
    angle *= -1
  else:
    image_flipped = image_light_changed
  return image_flipped, angle

def _load_data():
  colnames = ['center', 'left', 'right', 'steering', 
              'throttle', 'brake', 'speed']
  data = pandas.read_csv(os.path.join(DATA_PATH, 'driving_log.csv'), 
                         skiprows=[0], names=colnames)
  center_images =[v.strip() for v in data.center.tolist()]
  left_images = [v.strip() for v in data.left.tolist()]
  right_images = [v.strip() for v in data.right.tolist()]
  angles = data.steering.tolist()
  return center_images, left_images, right_images, angles
