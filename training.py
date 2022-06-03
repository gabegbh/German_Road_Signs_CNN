import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

def create_network():
    network = Sequential()
    network.add(Conv2D(input_shape=(32,32,3),filters=24,kernel_size=(3,3), activation="relu"))
    network.add(Conv2D(filters=24,kernel_size=(3,3), activation="relu"))
    network.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    network.add(Conv2D(filters=48, kernel_size=(3,3),  activation="relu"))
    network.add(Conv2D(filters=48, kernel_size=(3,3),  activation="relu"))
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

def display_metrics(history):
    """ plot loss and accuracy from keras history object """
    f, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history.history['loss'], linewidth=3)
    ax[0].plot(history.history['val_loss'], linewidth=3)
    ax[0].set_title('Loss', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].set_xlabel('Epoch', fontsize=16)
    ax[0].legend(['train loss', 'val loss'], loc='upper right')
    ax[1].plot(history.history['accuracy'], linewidth=3)
    ax[1].plot(history.history['val_accuracy'], linewidth=3)
    ax[1].set_title('Accuracy', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].set_xlabel('Epoch', fontsize=16)
    ax[1].legend(['train acc', 'val acc'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    training_data_gen = ImageDataGenerator(validation_split=0.2)
    training_data = training_data_gen.flow_from_directory(directory="GTSRB/Final_Training/Images",target_size=(32,32), subset='training')
    validation_data = training_data_gen.flow_from_directory(directory="GTSRB/Final_Training/Images",target_size=(32,32), subset='validation')

    opt = Adam(lr=0.0005)

    model = create_network()
    model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    history = model.fit(x=training_data, 
                        epochs=20, 
                        validation_data=validation_data)
    display_metrics(history)
