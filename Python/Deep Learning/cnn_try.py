# Convolutional Neural Networks - Computer Vision

# Image Classification

# Part 1 - Building the Convolutional Neural Network model

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing thr CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
# Using 32 feature detectors of size (3,3), input shape represents the dimensions of the images. 
# 64, 64 represents the 2D pixel values. 64 is used instead of the original value of 256 to reduce 
# computation and 3 represents the three arrays for red, blue and green. Here stride is 3*3

# Step 2 - Pooling
# Reducing the size of feature maps
# Here stride is 2*2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding Another convolutional layer (need not enter i/p shape as it it needed only for the first layer) 
# followed by a pooling layer
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3 - Flattening
# Putting all the numbers from the feature map into one same vector to be passed on to the ANN
# By convolution and pooling we keep the spatial structure of the feature
classifier.add(Flatten())

# Step 4 - Full Connection (Create hidden and output layers of ANN)
# Hidden Layer
classifier.add(Dense(units = 128,  activation = 'relu'))
# Output Layer
classifier.add(Dense(units = 1,  activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the Images
# image augmentation used to improve model performance without overfitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)