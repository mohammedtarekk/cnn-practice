import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import os
import cv2
import csv
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

trainpath = 'cnn3/train'
validatepath = 'cnn3'
TEST_DIR = 'cnn3/validate/'


trainBatches = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
        validation_split=0.2
        ).flow_from_directory(directory=trainpath,target_size=(224,224),classes=['non_autistic','autistic'],batch_size=10)

#testBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=testpath,target_size=(224,224),classes=['humans','zombies'],batch_size=10,shuffle=False)
valid = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=validatepath,target_size=(224,224),classes=['validate'],batch_size=10,shuffle=False)
print(valid.classes)
imgs, labels = next(valid)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    BatchNormalization(),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dropout(0.5),
    Dense(units=2, activation='softmax') #sigmoid
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) #loss = binar-crossentropy

if os.path.exists('model.h5'):
    model = load_model('./model.h5')
    print("kjas")
else:
    model.fit(x=trainBatches,
              steps_per_epoch=len(trainBatches),
              epochs=110,
              verbose=2
              )
    model.save('model.h5')




predict= model.predict(x=valid, steps=len(valid), verbose=0)

np.round(predict)
print(np.argmax(predict, axis=-1))

y = np.argmax(predict, axis=-1)[0]
print(y)

testing_results = []
i = 0
for img in tqdm(os.listdir(TEST_DIR)):
    testing_results.append([img, 0])
    i += 1

# write results in the file
with open('Submit.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Label'])
    writer.writerows(testing_results)

