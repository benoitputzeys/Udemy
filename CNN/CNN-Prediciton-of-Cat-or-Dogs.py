#Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
#pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D # Used to deal with images.
from keras.layers import MaxPooling2D # Used to add pooling layer
from keras.layers import Flatten # Used to create the large feature vector that will be the input to the network.
from keras.layers import Dense
from keras.models import model_from_json


# Here there is no data preprocessing because part of this preprocessing has been done manually:
# We have created 2 folders named cats and dogs respectively which already tell the computer what label the image is.
# The cats and dogs images are also in folders with the correct ratios of training to testing. (20/80)

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32 is the number of feature maps and 3x3 are the dimensions of the feature detector.
# Careful with the order in which you put down the integers. This is for Tensorflow backend.
# Other libraries require differnt orders as inputs.
# Because the images have different formats and different sizes and shapes, you have to standardise the input and
# You could use 64x64 or even higher such as 128x128 or 256x256 but this will entail much more required processing power and memory.
# The number 3 after 64x64 is to say that we are trying to predict color images.
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Now the huge vector is created that contains all the spatial information of the image.
classifier.add(Flatten())

# Step 4 - Full connection
# Number of 128 is mostly obtained from experimenting.
# Also common practice to pick a power of 2.
# For CNNs, you only have to add the hidden layer
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# Optimizer is the optimizer parameter to choose th stochastic gradient descent.
# The loss function is the parameter to choose the loss function.
# The metrics parameter chooses the performance metric. Accuracy is the most common one.
# For the loss function, if we have more than 2 outcomes, then the loss function will be categorical cross entropy.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator


# This first part is the image augmentation part where we apply several transformations like rescale,
# shearing the image, a random zoom we apply to the images and flipping the images horizontally.
# The image augmentation is done on the training set and

# Rescaling means that we will rescale all our pixel values between 0 and 1 because pixels take on values between 0 and 255.
# By rescaling them with the factor 1/255, you get a value between 0 and 1.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# on the test set as well
test_datagen = ImageDataGenerator(rescale = 1./255)


# Here you define from where you get the data. Because you work in a working directory, you do not have to write down the
# WHOLE path but only the path from the directory.
# Input the dimensions expected by the CNN. Looking above, we have specified 64x64
# 32 corresponds to the number of images that go through the CNN after which the weight will be updated.
# Class:mode = 'binary' because you have a binary outcome.
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'binary')

# Steps_per_epoch is the number of images in the training set. Remember: all the observations of the training set pass
# through the NN during each epoch.
# Validaton data corresponds to the test set on which we want to evaluate the performance of the CNN.
classifier.fit_generator(training_set,
                         epochs=15,
                         validation_data=test_set)

# # serialize model to JSON
# classifier_json = classifier.to_json()
# with open("classifier.json", "w") as json_file:
#      json_file.write(classifier_json)
#
# # serialize weights to HDF5
# classifier.save_weights("classifier.h5")
# print("Saved model to disk")
#
# # load json and create model
# json_file = open('classifier.json', 'r')
# loaded_classifier_json = json_file.read()
# json_file.close()
# loaded_classifier = model_from_json(loaded_classifier_json)
# # load weights into new model
# loaded_classifier.load_weights("classifier.h5")
# print("Loaded classifier from disk")
#
# # evaluate loaded model on test data
# loaded_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# score = loaded_classifier.evaluate(train_datagen)
# print("%s: %.2f%%" % (loaded_classifier.metrics_names[1], score[1]*100))


# Make a prediciton.
# Method Nr. 1

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # The new dimension corresponds to the batch: The functions of NN,
# like the predict function, cannot accept a single input by itself, like the image we have here.
 # It only accepts inputs in a batch. Even if the batch contains one input,the input must be in the batch,
 # and this new dimension that we are creating right nowcorresponds to the batch.
 # In general, we can have several batches of several inputs, and we can apply the predict method on that.
 # So for example, if for this new dimension we had two elements, that means that we would have
 # two batches containing single or several inputs.
 # Also the input into this new dimension is 1. So you have a numpy array of 1,64,64,3 with 3 denoting the RGB colors.
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

# Method Nr. 2
#
# import cv2
#
# img1 = cv2.imread('CNN/single_prediction/cat_or_dog_1.jpg')
# img1 = cv2.resize(img1,dsize = (64,64))
# img1 = np.expand_dims(img1, axis = 0)
#
# y_predict = classifier.predict_classes(img1)