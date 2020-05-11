import os
import random
import collections
import numpy as np
import pickle
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

import utility

class Dataset:
    
    def __init__(self, dataset_path):
        train, valid, test = self._load_data(dataset_path)
        # train set
        self._X_train = train['features']
        self._y_train = train['labels']
        assert(self._X_train.shape[0] == self._y_train.shape[0])
        # validation set
        self._X_valid = valid['features']
        self._y_valid = valid['labels']
        assert(self._X_valid.shape[0] == self._y_valid.shape[0])
        # test set
        self._X_test = test['features']
        self._y_test = test['labels']
        assert(self._X_test.shape[0] == self._y_test.shape[0])
        # ensure image dimensions are all the same across datasets
        assert(self._X_train.shape[1:] == self._X_valid.shape[1:] == self._X_test.shape[1:])
        self._classes = self._cal_classes()
        self._id_to_name = self._build_class_name_map()
        
    def get_train(self):
        return self._X_train, self._y_train
    
    def get_valid(self):
        return self._X_valid, self._y_valid
    
    def get_test(self):
        return self._X_test, self._y_test
    
    def get_size_of_train(self):
        return self._y_train.shape[0]
    
    def get_size_of_valid(self):
        return self._y_valid.shape[0]
    
    def get_size_of_test(self):
        return self._y_test.shape[0]
    
    def get_size_of_image(self):
        # format: (width, height, #channels)
        return self._X_train.shape[1:]
    
    def get_classes(self):
        return self._classes
    
    def get_num_of_classes(self):
        return len(self._classes)
    
    def get_class_name(self, class_id):
        assert(class_id in self._id_to_name)
        return self._id_to_name[class_id]
    
    def _cal_classes(self):
        # assume all three datasets have the same number of classes
        classes_a = sorted(np.unique(self._y_train))
        classes_b = sorted(np.unique(self._y_valid))
        classes_c = sorted(np.unique(self._y_test))
        assert(classes_a == classes_b == classes_c)
        return classes_a
    
    def _build_class_name_map(self):
        id_to_name = {}
        for class_id, sign_name in np.genfromtxt("signnames.csv", skip_header=1,
                dtype=[("class_id", np.uint64), ("sign_name", "S80")], delimiter=","):
            id_to_name[class_id] = sign_name.decode()
        return id_to_name
        
    def _load_data(self, dataset_path):
        get_path = lambda fname: os.path.join(dataset_path, fname)
        with open(get_path("train.p"), mode='rb') as f:
            train = pickle.load(f)
        with open(get_path("valid.p"), mode='rb') as f:
            valid = pickle.load(f)
        with open(get_path("test.p"), mode='rb') as f:
            test = pickle.load(f)
        return train, valid, test


class Classifier:
    
    def __init__(self, dataset, mu=0, sigma=0.01, keep_prob=0.5, rate=0.0008, epochs=50, batch_size=128, target_size=400):
        # preprocess raw data
        data, self._y_train = dataset.get_train()
        #self._X_train = self._preprocess(data)
        #self._X_train, self._y_train = self._resample_dummy(
        #    self._X_train, self._y_train, dataset.get_classes(), 1000)
        data, label = dataset.get_train()
        self._X_train, self._y_train = self._preprocess_training_data(data, label, dataset.get_classes(), target_size)
        
        data, self._y_valid = dataset.get_valid()
        self._X_valid = self._preprocess(data)
        data, self._y_test = dataset.get_test()
        self._X_test = self._preprocess(data)
        
        # data description
        self._image_shape = self._X_train[0].shape
        self._n_classes = dataset.get_num_of_classes()
        self._n_channels = 1
        
        # hyperparameters
        self._mu = mu
        self._sigma = sigma
        self._keep_prob = keep_prob
        self._rate = rate
        self._epochs = epochs
        self._batch_size = batch_size 
        
        # setup tensorflow environment
        self._setup_tf_env()

    def get_train(self):
        return self._X_train, self._y_train
    
    def get_valid(self):
        return self._X_valid, self._y_valid
    
    def get_test(self):
        return self._X_test, self._y_test

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self._batch_size):
            batch_x, batch_y = X_data[offset: offset+self._batch_size], y_data[offset: offset+self._batch_size]
            accuracy = sess.run(self._accuracy_operation, feed_dict={self._X_knob: batch_x, 
                                                                     self._y_knob: batch_y, 
                                                                     self._kp_knob: 1})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(self._X_train)
            print("Training starts\n")
            for i in range(self._epochs):
                self._X_train, self._y_train = shuffle(self._X_train, self._y_train)
                for offset in range(0, num_examples, self._batch_size):
                    end = offset + self._batch_size
                    batch_x, batch_y = self._X_train[offset:end], self._y_train[offset:end]
                    sess.run(self._training_operation, feed_dict={self._X_knob: batch_x, 
                                                                  self._y_knob: batch_y, 
                                                                  self._kp_knob: self._keep_prob})
                train_accuracy = self.evaluate(self._X_train, self._y_train)    
                validation_accuracy = self.evaluate(self._X_valid, self._y_valid)
                print("EPOCH {}".format(i+1))
                print("Train set Accuracy = {:.3f}".format(train_accuracy))
                print("Validation set Accuracy = {:.3f}\n".format(validation_accuracy))
            test_accuracy = self.evaluate(self._X_test, self._y_test)
            print("Test Set Accuracy = {:.3f}".format(test_accuracy))
            tf.train.Saver().save(sess, './lenet')
            
    def test(self, Xd, yd):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('./lenet.meta')
            saver.restore(sess, "./lenet")
            test_accuracy = self.evaluate(Xd, yd)
            print("Test Set Accuracy = {:.3f}".format(test_accuracy))
            
    def get_prediction_details(self, Xd, yd):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('./lenet.meta')
            saver.restore(sess, "./lenet")
            sign_ids = sess.run(tf.argmax(self._logits, 1), feed_dict={self._X_knob: Xd, 
                                                                       self._y_knob: yd, 
                                                                       self._kp_knob: 1})
            top_k = sess.run(tf.nn.top_k(tf.nn.softmax(self._logits), k=5), 
                             feed_dict={self._X_knob: Xd, 
                                        self._y_knob: yd, 
                                        self._kp_knob: 1})
            return sign_ids, top_k
        
    def _LeNet(self, x, keep_prob):    
        # Layer 1: Convolutional. Input = 32x32xn_channels. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, self._n_channels, 6), mean=self._mu, stddev=self._sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        # Layer 1: Activation.
        conv1 = tf.nn.relu(conv1)
        # Layer 1: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=self._mu, stddev=self._sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        # Layer 2: Activation.
        conv2 = tf.nn.relu(conv2)
        # Layer 2: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Layer 2: Flatten. Input = 5x5x16. Output = 400.
        fc0   = flatten(conv2)
        fc0   = tf.nn.dropout(fc0, keep_prob=keep_prob)
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self._mu, stddev=self._sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        # Layer 3: Activation.
        fc1    = tf.nn.relu(fc1)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self._mu, stddev=self._sigma))
        fc2_b  = tf.Variable(tf.zeros(84))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        # Layer 4: Activation.
        fc2    = tf.nn.relu(fc2)
        # Layer 5: Fully Connected. Input = 84. Output = n_class.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, self._n_classes), mean=self._mu, stddev=self._sigma))
        fc3_b  = tf.Variable(tf.zeros(self._n_classes))
        logits = tf.matmul(fc2, fc3_W) + fc3_b
        return logits
        
    def _preprocess(self, rgb_images):        
        # turn into grayscale image
        gray_images = np.sum(rgb_images/3.0, axis=3, keepdims=True)
        # normalize the image
        normalized = (gray_images - 128.0) / 128.0
        # debug
        # print(rgb_images.shape, "->", normalized.shape)
        return normalized
    
    def _preprocess_training_data(self, rgb_images, labels, classes, target_sample_size):
        total = 0
        counts = collections.Counter(labels)
        image_indices = collections.defaultdict(list)
        for i, label in enumerate(classes):
            if counts[label] < target_sample_size:
                image_indices[label].append(i)
                total += target_sample_size - counts[label]
        
        print("Original data shape ", rgb_images.shape, labels.shape, total)
        gray_images = np.sum(rgb_images/3.0, axis=3, keepdims=False)
        added_images = np.empty(shape=(total,) + gray_images[0].shape, dtype=float)
        added_labels = np.empty(shape=(total,), dtype=int)
        index = 0
        for label in classes:
            if counts[label] < target_sample_size:
                options = image_indices[label]
                for i in range(target_sample_size - counts[label]):
                    added_images[index] = utility.generate_noisy_image(
                        gray_images[options[random.randrange(len(options))]])
                    added_labels[index] = label
                    index += 1
                #print(target_sample_size - counts[label], label, index)   
        norm_images1 = (gray_images - 128.0) / 128.0
        norm_images2 = (added_images - 128.0) / 128.0
        all_images = np.concatenate((norm_images1, norm_images2))[:, :, :, np.newaxis]
        all_labels = np.concatenate((labels, added_labels))
        
        print("Preprocessed data shape: ", all_images.shape, all_labels.shape)
        return shuffle(all_images, all_labels)
    
    def _setup_tf_env(self):
        # placeholder for a batch of input images
        self._X_knob = tf.placeholder(tf.float32, (None,) + self._image_shape)
        # placeholder for a batch of output labels
        self._y_knob = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(self._y_knob, self._n_classes)
        # placeholder for dropping
        self._kp_knob = tf.placeholder(tf.float32)
        # training operation
        self._logits = self._LeNet(self._X_knob, self._kp_knob)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=self._logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = self._rate)
        self._training_operation = optimizer.minimize(loss_operation)
        # accuracy operation
        correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(one_hot_y, 1))
        self._accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ######################## Junk code ###########################
    
    def _resample_dummy(self, images, labels, classes, target_size):
        return shuffle(images, labels)
    
    def _resample_to_target_size(self, images, labels, classes, target_size):
        """
        Equalize the sample size of each class to the max sample size observed in the dataset
        """
        n_classes = len(classes)
        total = n_classes * target_size
        adjusted_images = np.empty(shape=(total,) + images[0].shape, dtype=int)
        adjusted_labels = np.empty(shape=(total,), dtype=int)
        # Get the actual data
        indices = [0] * n_classes
        for i, label in enumerate(labels):
            index = indices[label]
            if index >= target_size:
                continue
            adjusted_images[index + label * target_size] = images[i]
            adjusted_labels[index + label * target_size] = label
            indices[label] += 1
        # Duplicate data if sample size is less than target size
        for label in classes:
            base = label * target_size
            actual_length = indices[label]
            for i in range(base + actual_length, base + target_size):
                adjusted_images[i] = adjusted_images[base + random.randrange(actual_length)]
                adjusted_labels[i] = label
        return shuffle(adjusted_images, adjusted_labels)
    
    def _resample_at_least_target_size(self, images, labels, classes, target_size):
        """
        Equalize the sample size of each class to the max sample size observed in the dataset
        """
        # calculate the total number of samples as well as the starting index 
        # for each label
        total = 0
        indices = []
        for _, count in sorted(collections.Counter(labels).items()):
            indices.append(total)
            total += count if count >= target_size else target_size
        indices.append(total)
        start_indices = indices[:]
        # prepare the output 
        adjusted_images = np.empty(shape=(total,) + images[0].shape, dtype=int)
        adjusted_labels = np.empty(shape=(total,), dtype=int)
        # Get the actual data
        for i, label in enumerate(labels):
            index = indices[label]
            adjusted_images[index] = images[i]
            adjusted_labels[index] = label
            indices[label] += 1
        #print(start_indices, '\n', indices)
        # Duplicate data if sample size is less than target size
        for label in classes:
            if start_indices[label + 1] == indices[label]:
                continue
            base = start_indices[label]
            actual_length = indices[label] - base
            for i in range(indices[label], start_indices[label + 1]):
                adjusted_images[i] = adjusted_images[base + random.randrange(actual_length)]
                adjusted_labels[i] = label
        return shuffle(adjusted_images, adjusted_labels)
