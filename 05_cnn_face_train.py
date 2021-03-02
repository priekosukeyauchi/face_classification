#python 04_cnn_face_train.py model/final_model.h5
import os
import tensorflow.keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import time
import sys


img_width, img_height = 64, 64

batch_size = 32
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
nb_patience = 10
nb_filter2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 4
lr = 0.0004
epochs = 200

train_data_dir = "dataset/03_face_inflated/"
validation_data_dir = "dataset/04_validation/"


model = Sequential()
model.add(Conv2D(nb_filters1, conv1_size, conv1_size, padding ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters2, conv2_size, conv2_size, padding ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

target_dir = 'model/'
if not os.path.exists(target_dir):
  os.makedirs(target_dir)
model.save('model/final_model.h5')
model.save_weights('model/weights.h5')


plt.xlabel("time step")
plt.ylabel("accuracy")

plt.ylim(0, max(history.history["accuracy"]))

accuracy, = plt.plot(history.history["accuracy"], c="#56B4E9")


plt.legend([accuracy], ["accuracy"])

plt.show()

plt.savefig("cnn_face_train_figure.png")
