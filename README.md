# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goal of this project is to teach the computer to drive car based on deep learning 
and convolutional neural network. Using a simulator, we can obtain driving data recording
images captured by three cameras (right, center, and left camera facing the front of the
car in a line), steering angle, and so on. Then we extract the useful
data (images and steering angles) and preprocess them. Using these preprocessed data, 
we train a model, based on Keras deep learning framework, to predict the steering angle
given the images captured by the center cameras. With the smart prediction model, we can 
drive the car in a simulated track smoothly. 


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* data.py, a module processing raw data obtained from simulation 
* model.py, a module training the prediction model
* train, a script training the prediction model
* drive, a script driving car using our prediction model
* model.json, a data file containing saved model
* model.h5, a data file containing saved model weights

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously 
around the track by executing 

```sh
./drive PATH_FOR_SAVED_IMAGES
```

Note we use './simulation' as our path for the saved images.

#### 3. Submission code is usable and readable

The file data.py, model.py, and train contain the code for training and saving the convolution neural network. The 
file shows the pipeline I used for training and validating the model, and it contains comments 
to explain how the code works.

### Data Processing

The post [You don’t need lots of data!](https://medium.com/@fromtheast/you-dont-need-lots-of-data-udacity-behavioral-cloning-6d2d87316c52)
is the basis for our data processing. It claims that the existing data is already enough for training
a good model supposing we do a good work with data augmentation, which eliminates the need to add
more data with driving simulation. Following this, our data preprocessing consists of 5 steps as follows.

First, load and extract useful data. We load the data metadata from .csv file and extract only the
names of left/right/center images and the steering angles, where the images would be the input
of our model and steering angels would be the labels/output. 

Second, split data for training and validation. We shuffle the center images and sterring angles and
split into 80/20 for training and validation.

Third, add more data describing how a car going to be away from the track gets back to the center
of the track, i.e., we need to have more data with bigger steering angles. The left/right cameras 
view the point of interest in either larger or smaller angle compared with the center camera does 
depending on whether the car of going forward or turning left/right. This facts makes images 
captured by them a perfect fit for describing the recovery of a car on track. We set the steering 
angles offset (angle difference between left/right camera and center camera) to 0.27. We ensure that 
the amount of such recovery data with large angle is at least 500.  

Forth, inject noise to the image data:
- Adjust the brightness of the image randomly, making the model agnostic to the environment brightness.
- Flip the images in the left/right direction randomly, making the model agnostic to the track direction.
- Cut off the upper and lower part of the images as they are just noise to our model
- Resize the image to 64x64x3, reducing computation complexity of the model 

Fifth, use python generator in the data generationm, incurring less memory footprint. Most importantly,
to make full use of available data, we inject noise to each image in the batch data generation.    

### Model Architecture and Training Strategy

After browsing through multiple existing projects, we find the most, if not all, use the model detailed 
in [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
by Nvidia with only a slight difference on the data preprocessing and parameter choice. The 
[project by An Nguyen](https://github.com/ancabilloni/SDC-P3-BehavioralCloning) did also the same. We use 
the prameter choice in this project in our implementation.  

Our model takes 64x64x3 images. Its training pipeline consists of three steps: 
- It normalizes the image pixel value range to -0.5 and 0.5. 
- It first processes with 3 convolution layers with filter size 5x5, stride 2x2, and filter depth 24, 36 and 48, and then
with 2 convolution layers with filter size 3x3, stride 2x2, and filter depth 64. Note each convolution layer is followed by
a Relu activation for adding non-linearty to the model. 
- It then flattens the last layer from the above and processes with 4 fully-connected layers (80, 40, 16, 10) 
before the final output (1). We apply a 50% dropout after each of the first 3 fully-connected layer to 
avoid overfitting. Also we apply L2 weight regularization in each layer to avoid overfitting.  

With the above model, we train the model with batch size 128 and 3 epochs (It is hard to us to train the model
longer as each epoch takes about 1 hour and the workspace would time out every time it runs too long. We tried 
multiple times, but the timeout behavior of the workspace failed me everytime. Also we tried do experiments on
our personal PC, but failed installing the simulator correctly. Asked help in our discussion channel, but got 
little help there). 

### Self-driving Test

We validate our model via simulation. The simulation demonstrates that our car navigate safely in the center 
of the track as a human driver. The demo video is posted on youtube as below.  

[Self-driving Demo](https://youtu.be/LFKr8J-Yluw)

### Summary

Our model is verified to work with this track, but is not good enough for the more challenging track due to 
the underfitting problem. We don't train with enough epochs as the platform is a little fragile on usability. 
Given enough time, a better model should be generated to deal with the more challenging track. 
