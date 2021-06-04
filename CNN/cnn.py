import os
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_AUT_DIR = r"train\autistic"
TRAIN_NON_DIR = r"train\non_autistic"
TEST_DIR = "test"
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = "autism-disorder-cnn"

def create_label(imgName):
    word_label = imgName.split('.')[0][0:3]
    if word_label == 'aut':
        return np.array([1, 0])
    elif word_label == 'non':
        return np.array([0, 1])

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_AUT_DIR)):
        path = os.path.join(TRAIN_AUT_DIR, img)
        img_data = cv2.imread(path).astype(np.float32)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])

    for img in tqdm(os.listdir(TRAIN_NON_DIR)):
        path = os.path.join(TRAIN_NON_DIR, img)
        img_data = cv2.imread(path).astype(np.float32)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

if os.path.exists('train_data.npy'):
    train_data = np.load('train_data.npy', allow_pickle=True)
else:
    train_data = create_train_data()

# Split data
train = train_data
x_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = [i[1] for i in train]

# Real-time image preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# build network
#tf.compat.v1.reset_default_graph()
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], data_augmentation=img_aug, data_preprocessing=img_prep)

# network = conv_2d(network, #feature_maps, filterSize, activation='relu')
network = conv_2d(network, 32, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 128, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 32, 5, activation='relu')
network = max_pool_2d(network, 5)

network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)

network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR)

model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit(x_train, y_train, n_epoch=100, snapshot_step=500, shuffle=True, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

# testing
testing_results = []
for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path).astype(np.float32)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        prediction = model.predict(img_data)[0]
        print(f"autistic: {prediction[0]}, non_autistic: {prediction[1]}")
        if prediction[0] > prediction[1]:
            testing_results.append([img, 1])
        else:
            testing_results.append([img, 0])

# write results in the file
with open('Submit.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Label'])
    writer.writerows(testing_results)
