import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D

from tensorflow.keras import callbacks


DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 100

train_data_path = 'C:/Users/dell/Desktop/AIrizal/data/train'
validation_data_path = 'C:/Users/dell/Desktop/AIrizal/data/validation'

"""
Parameters
"""

img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 5
validation_steps = 5
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 11
lr = 0.0004

model = Sequential()
model.add(Conv2D(nb_filters1, conv1_size, conv1_size, padding ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters2, conv2_size, conv2_size, padding ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))

model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
log_dir = 'C:/Users/dell/Desktop/AIrizal/tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]


hist = model.fit_generator(
    train_generator,
    steps_per_epoch = samples_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    callbacks = cbks,
    validation_steps = validation_steps)

target_dir = 'C:/Users/dell/Desktop/AIrizal/models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('C:/Users/dell/Desktop/AIrizal/models/model.h5')
model.save_weights('C:/Users/dell/Desktop/AIrizal/models/weights.h5')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.plot (hist.history['loss'], color = 'blue', label = 'train')
plt.plot (hist.history['val_loss'], color = 'orange', label = 'train')
plt.grid(True)
plt.title("Train & test loss with epochs\n", fontsize = 16)
plt.xlabel ("Training epochs", fontsize=12)
plt.ylabel ("Train & test loss", fontsize=12)
plt.show()

plt.plot (hist.history['acc'], color = 'blue', label = 'train')
plt.plot (hist.history['val_acc'], color = 'orange', label = 'train')
plt.grid(True)
plt.title("Train & test accuracy with epochs\n", fontsize=16)
plt.xlabel ("Training epochs", fontsize=12)
plt.ylabel ("Train & test loss", fontsize=12)
plt.show()
