## Traffic Sign Recognition

The goal of this project is to build a CNN model to identify traffic sign. This project can be broken down into following steps. 

- Summarize and Explore Dataset
- Design and Test a Model Architecture
- Test the Model on New Images

In this report, we will first summarize our implementation. Then we will discuss the challenges and solutions in each of the above steps. In the end, we summarize this project. 

[//]: # (Image References)

[all_signs]: ./resources/AllSigns.png "All Traffic Signs"
[label_distribution]: ./resources/LabelDistribution.png "Label Distribution of Training Set"
[transformed]: ./resources/Transformed.png "Random Transformation Demo"
[lanet]: resources/lanet.png "Classical LaNet"
[new]: resources/new.png "New Data"
[new1]: new_images/1-Speed-limit-30-km-h.jpg "1-Speed-limit-30-km-h"
[new2]: new_images/11-Right-of-way-at-the-next-intersection.jpg "11-Right-of-way-at-the-next-intersection"
[new3]: new_images/12-Priority-road.jpg "12-Priority-road"
[new4]: new_images/14-Stop.jpg "14-Stop"
[new5]: new_images/15-No-vehicles.jpg "15-No-vehicles"

---
### 0 Implementation Summary

To make a clear and concise description of our work. We organize most of our code into two seperate modules. 

- Core, it contains all of our critical code, i.e., Dataset and Classifier
- Utility, it contains our utility functions for illustrating our data

On top of these two modules, the step-by-step description and code execution is given at [project code](Traffic_Sign_Classifier.ipynb)

---
### 1 Summarize and Explore Dataset

This part consists of two steps: 
* Load the data set
* Explore, summarize and visualize the data set

Below is a summary of our dataset. 
- Number of training examples =  34799
- Number of validation examples =  4410
- Number of testing examples =  12630
- Image data shape =  (32, 32, 3)
- Number of classes (labels) =  43

Below we plot one image for each type of labels. 
![alt text][all_signs]

Also we plot the label distribution of the traing dataset. Note we also did this for the validation and training dataset and found all of them presents similar label distribution. 
![alt text][label_distribution]

---
### 2 Design and Test a Model Architecture

### 2.1 Pre-processing

Our preprocessing consists of two steps:

- We convert source RGB images into grayscale images as lines and shapes provide enough information for traffic sign identification

- We normalize the value of grayscale images to (-1, 1) to enable fast and stable convergence -- without it the value range of features might vary and in our learning the correction applied in each dimension might vary. 

- We generate fake data for labels if the sample size of the label is less than a user specified bar to avoid labels with significantly large sample size being biased. We generate fake data via following steps: (1) select an image of the label randomly; (2) roatate the image by a random degree (-10, 10); (3) blur the image with gaussian smoothing with a random kernel size (1, 3); (4) zoom out the image to size (32 + inc, 32 + inc) with inc being a random integer value no less than 5. The reason we have to add noise into the image is that this would increase our sample size yet wihout making the model biased towards the duplicated images. (Borrowed some idea from project https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project, but here we )

We cannot generate too much fake data (like 1000 samples per label) as this will lead to worse performance compared with the situation without doing so. In our experiements, we use 400 as our best choice. Below illustrate our transformation. On the left is the original image from our dataset, and on the right are 4 randomly transformed images. 
![alt text][transformed]

### 2.2 Model Architecture

![alt text][lanet]

We use the classical LaNet network, consisting of 2 convolutional layers and 3 fully connected layers. Note the only difference we have compared with the origianl LaNet is that we add a dropout immediately after the 2nd convolutional layer. Below is a step-by-step breakdown of the model architecture with input being 32x32x1 grayscale images. 

- Convolution (kernel size: 5x5, stride: 1x1): output 28x28x6
- Relu
- Max pooling (kernel size: 2x2, stride: 2x2): output 14x14x6
- Convolution (kernel size: 5x5, stride: 1x1): output 10x10x16
- Relu
- Max pooling (kernel size: 2x2, stride: 2x2): output 5x5x16
- Flatten: output 400
- Dropout: output 400 * keep_prob
- Fully connected: output 120
- Relu
- Fully connected: output 84
- Relu
- Fully connected: output 43 (number of classes of our data)

### 2.3 Model Training

We use Adam optimizer to minimize the loss function, which is a known default method better than stochastic gradient discent. We use the following hyperparameters in the training. 

- Learning rate: 0.00075
- Number of epochs: 80
- Batch size: 128
- Keep probability for dropouts: 0.5
- Standard deviation for weights initialization: 0.01

### 2.4 Validate Model

We evaluate model after each epoch. At 40th epoch, our validation accuracy stays over **94%** steadily. The highest validation accuracy observed is 96%, but this caused two misprediction in our new test images due to overfitting (different hyperparameter setting). 

### 2.5 Test Model

Our test dataset accuracy is **93.7%**. 

### 2.6 Our Road

We starts from the basic LaNet with an added dropouts. For the parameter tuning, we start from the values used in the two projects below. 

https://github.com/ser94mor/traffic-sign-classifier

https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project


Afterwards, we add fake data by simply duplicating data. But simply duplicating data makes the model biased towards the duplicated data. To avoid this issue, we used a more advanced way generating fake data, inspired by the 2nd project (the way we generate data are slightly different). The remaining then is just a boring trial-and-error process. 

---
## 3 Test a Model on New Images

We download the 5 new test images from https://github.com/ser94mor/traffic-sign-classifier. Each of the image name contains the label the image belongs to. The traffic signs we include are given as below:

- Speed limit (30km/h): label 1
- Right-of-way at the next intersection: label 11
- Priority road: label 12
- Stop: label 14
- No vehicles: label 15

![alt text][new]

### 3.1 Preprocess Data

We did the following to the preprocessing:

- resize the image to our required sizes (32x32)
- preprocess the images with grayscale conversion and normalization

### 3.2 Predict Label and Analyze Performance

We ran our code 3 times. In two runs, the accuracy is 100%; In the other run, the accuracy is 80% where the image of  label-1 was misidentified. 

### 3.3 Output Top 5 Softmax Probabilities For Each Image

Below is the the top 5 softmax probabilities for each image in our last run. As we can say, our model can identify the right label for these images with high confidence, considering the probability of the winning label is very close to 1.0. 

[[  9.73632812e-01   2.43905913e-02   1.67271239e-03   1.69229505e-04
    4.85454948e-05]
    
 [  9.99999762e-01   1.59818441e-07   9.45678238e-08   1.11757503e-09
    3.90905580e-12]
    
 [  1.00000000e+00   2.13622577e-08   3.92097105e-10   1.31977929e-10
    6.15066956e-11]
    
 [  9.16332483e-01   7.57231265e-02   7.85820931e-03   8.41379224e-05
    1.28613033e-06]
    
 [  8.32073331e-01   1.43308014e-01   7.06136692e-03   6.05386030e-03
    4.36712196e-03]]

---
## References

https://github.com/ser94mor/traffic-sign-classifier

https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project
