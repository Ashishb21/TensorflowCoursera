import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import shutil
from numpy import load
from scipy.io import loadmat


base_dir='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir=base_dir+'Data'
dataset_path=os.path.join(data_dir,'SVHN/')
#os.mkdir(os.path.join(data_dir,'SVHN'))
#shutil.move('/Users/ashishbansal/Downloads/test_32x32.mat',dataset_path)
#shutil.move('/Users/ashishbansal/Downloads/train_32x32.mat',dataset_path)
# os.mkdir(dataset_path+'test')
# os.mkdir(dataset_path+'train')

# zipped=zipfile.ZipFile(dataset_path+'test.tar.gz',mode='r')
# zipped.extractall(dataset_path+'test/')
# zipped=zipfile.ZipFile(dataset_path+'train.tar.gz',mode='r')
# zipped.extractall(dataset_path+'train/')
# zipped.close()
#shutil.unpack_archive(dataset_path+'test.tar.gz',dataset_path)

train=loadmat(dataset_path+'train_32x32.mat')
test=loadmat(dataset_path+'test_32x32.mat')
#print(train.keys())
train_images=train['X']
train_label=train['y']
test_images=test['X']
test_labels=test['y']

print(train_images.shape)
print(train_label.shape)

train_images=train_images.transpose((3,0,1,2))
test_images=test_images.transpose((3,0,1,2))
print(train_images.shape)
print(test_images.shape)

train_images =train_images[:,:,:,]
print(train_images[0].shape)

tf.reduce_mean()
#plot_images(train_images)
train_images=train_images/255.0
test_images=test_images/255.0
print(np.unique(train_label))

train_labels_encoding= tf.keras.utils.to_categorical(train_label)
test_labels_encoding=tf.keras.utils.to_categorical(test_labels)
print(train_labels_encoding.shape)
print(test_labels_encoding.shape)
print(train_labels_encoding[3])

def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                     input_shape=(32,32,3), name='conv_1'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_1'))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='conv_2'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(8, 8), name='pool_1'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='dense_2'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#model=get_model()
#print(model.summary())

def get_early_stopping():
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, mode='max')
    return earlystopping

def get_checkpoint_every_epoch():
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_every_epoch/checkpoint_{epoch:002d}'
                                                         , save_freq='epoch', save_weights_only=True)
    return modelcheckpoint

def get_checkpoint_best_only():

    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_best_only/checkpoint',
                                                         monitor='val_acc', save_best_only=True,
                                                         mode='max', save_weights_only=True)
    return checkpoint_best

checkpoint_every_epoch = get_checkpoint_every_epoch()
checkpoint_best_only = get_checkpoint_best_only()
early_stopping = get_early_stopping()

callbacks = [checkpoint_every_epoch, checkpoint_best_only, early_stopping]
#history=model.fit(train_images, train_labels_encoding, epochs=10, validation_data=(test_images, test_labels_encoding), callbacks=callbacks)
#pd.DataFrame(history.history).plot(figsize=(5,3))
#plt.show()