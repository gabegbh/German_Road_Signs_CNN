import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import numpy as np

def create_network():
    network = Sequential()
    network.add(Conv2D(input_shape=(32,32,3),filters=12,kernel_size=(3,3), activation="relu"))
    network.add(Conv2D(filters=12,kernel_size=(3,3), activation="relu"))
    network.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    network.add(Conv2D(filters=24, kernel_size=(3,3),  activation="relu"))
    network.add(Conv2D(filters=24, kernel_size=(3,3),  activation="relu"))
    network.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # network.add(Conv2D(filters=256, kernel_size=(3,3),  activation="relu"))
    # network.add(Conv2D(filters=256, kernel_size=(3,3),  activation="relu"))
    # network.add(Conv2D(filters=256, kernel_size=(3,3),  activation="relu"))
    # network.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # network.add(Conv2D(filters=512, kernel_size=(3,3),  activation="relu"))
    # network.add(Conv2D(filters=512, kernel_size=(3,3),  activation="relu"))
    # network.add(Conv2D(filters=512, kernel_size=(3,3),  activation="relu"))
    # network.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # network.add(Conv2D(filters=512, kernel_size=(3,3),  activation="relu"))
    # network.add(Conv2D(filters=512, kernel_size=(3,3),  activation="relu"))
    # network.add(Conv2D(filters=512, kernel_size=(3,3),  activation="relu"))
    # network.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    network.add(Flatten())
    network.add(Dense(120, activation='relu'))
    network.add(Dense(84, activation='relu'))
    network.add(Dense(43, activation='softmax'))
    
    return network

if __name__ == '__main__':
    training_data_gen = ImageDataGenerator(validation_split=0.2)
    training_data = training_data_gen.flow_from_directory(directory="GTSRB/Final_Training",target_size=(32,32), subset='training')
    validation_data = training_data_gen.flow_from_directory(directory="GTSRB/Final_Training",target_size=(32,32), subset='validation')

    opt = Adam(lr=0.01)

    model = create_network()
    model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    model.summary()
