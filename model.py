from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, MaxPooling2D, Activation
from math import ceil
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

samples = []

# Load the training data
with open('./OutData4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Configure adding flipped image data for training
addFlipped = True

# Function to flip both the image on the vertical axis and negate the angle
def flipData(image, angle):
    image_flipped = np.fliplr(image)
    angle_flipped = -angle

    return image_flipped, angle_flipped


# Extracts images
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './' + batch_sample[0].split('/')[-3] + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                # Read in images from center, left and right cameras
                img_left = cv2.imread('./' + batch_sample[1].split('/')[-3] + '/IMG/'+ batch_sample[1].split('/')[-1])
                img_right = cv2.imread('./' + batch_sample[2].split('/')[-3] + '/IMG/'+ batch_sample[2].split('/')[-1])
                
                # Add images and angles to data set
                images.append(img_left)
                images.append(img_right)
                angles.append(steering_left)
                angles.append(steering_right)
                
                if addFlipped == True:
                    image_flipped, angle_flipped = flipData(center_image, center_angle)
                    images.append(image_flipped)
                    angles.append(angle_flipped)
                    
                    image_flipped, angle_flipped = flipData(img_left, steering_left)
                    images.append(image_flipped)
                    angles.append(angle_flipped)
                    
                    image_flipped, angle_flipped = flipData(img_right, steering_right)
                    images.append(image_flipped)
                    angles.append(angle_flipped)
                    
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Cropping 50 pixels from the top if the image and 20 from the bottom: 160 - 50 - 20 = 90
ch, row, col = 3, 90, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
            steps_per_epoch=ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)

model.save('model.h5')