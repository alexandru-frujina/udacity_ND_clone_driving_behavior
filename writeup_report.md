# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_resources/NVIDIA-PilotNet-architecture.png "Nvidia PilotNet architecture"
[image2]: ./OutData4/IMG/center_2019_12_05_20_01_26_369.jpg "Training data sample center camera"
[image3]: ./OutData4/IMG/left_2019_12_05_20_01_26_369.jpg "Training data sample left camera"
[image4]: ./OutData4/IMG/right_2019_12_05_20_01_26_369.jpg "Training data sample right camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture used was the one suggested by Nvidia:

![alt text][image1]

The final architecture contains MaxPooling, Dropout and RELU activation layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 (RGB image)  						| 
| Cropping2D         	| Output: 320x90x3 (RGB image)  				| 
| Lambda         		| Normalize image between -1 and 1  			| 
| Convolution2D 5x5   	| 2x2 stride, valid padding, 24 neurons 		|
| RELU					|												|
| Convolution2D 5x5   	| 2x2 stride, valid padding, 36 neurons 		|
| RELU					|												|
| Convolution2D 5x5   	| 2x2 stride, valid padding, 48 neurons 		|
| RELU					|												|
| Convolution2D 3x3   	| 2x2 stride, valid padding, 64 neurons 		|
| RELU					|												|
| MaxPooling			|												|
| Convolution2D 3x3   	| 2x2 stride, valid padding, 64 neurons 		|
| RELU					|												|
| Flatten				|												|
| Dense    				| 100 neurons 									|
| RELU					|												|
| Dropout				| 25%											|
| Dense    				| 50 neurons 									|
| RELU					|												|
| Dense    				| 10 neurons 									|
| RELU					|												|
| Dense    				| 1 output neuron 								|

The input image is first cropped in order to remove unimportant information: the hood of the car and the upper part that mostly contains the sky. The next step is the normalization from (0, 255) to (-1.0, 1.0). The following layes consist of a series of convolutional layers and fully connected layers with RELU activation functions, MaxPooling and Dropout layers in-between. The Dropout layer has the purpose of decreasing overfitting of the model and the RELU layers have the purpose of introducing non-linearity in the model.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 109).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. As expected the way the car is driven on the track impacts how the car is driven autonomously by the model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I did two laps of the track and recorded once more some of the more difficult turns.

I also augumented the data by using images from all 3 cameras as well as flip the images and reverse the angle.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to fine-tune the model proposed by Nvidia.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat overfitting, I modified the model to also use a Dropout layer.

The model itself went through some iterations to choose different numbers of the neurons in the fully-connected layer, the Dropout rate, adding RELU activation layers. Out of the entire train data the 20% was used for the validation of the model while the restof 80% was used for the training itself. The training data was given in batches of 32

On the first attempts the car went off-track when encountering more difficult curves. What helped mostly was the Dropout rate value, the training data and the number of epochs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 95-114) consisted of a convolution neural network with the following layers and layer sizes as in the table above.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

The left and right camera images follow:

![alt text][image3]![alt text][image4]

After two laps of recording a third lap was done but only with the more difficult curves recorded.

To augment the data sat, I also flipped images and angles thinking that this would prevent overfitting.

After the collection process, I had 4460 * 3 number images (4460 of each left, center and right). I then preprocessed this data by in the model pipeline by cropping and normalizing the image to -1, 1.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that the car drives worse when increasing the number of epochs.
