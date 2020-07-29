import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import pdb
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

rootdir=os.path.abspath(os.getcwd())
projectdir,_=os.path.split(rootdir)
datadir=os.path.join(projectdir,'Data')



def load_data(file):
    zip=zipfile.ZipFile(file=file,mode='r')
    zip.extractall(datadir)
    zip.close()

#load_data(datadir+'/cats-and-dogs.zip')

print(len(os.listdir(datadir+'/PetImages/Cat')))
print(len(os.listdir(datadir+'/PetImages/Dog')))

CAT_SOURCE_DIR = datadir+"/PetImages/Cat/"
TRAINING_CATS_DIR = datadir+"/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = datadir+"/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = datadir+"/PetImages/Dog/"
TRAINING_DOGS_DIR = datadir+"/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = datadir+"/cats-v-dogs/testing/dogs/"

# os.mkdir(datadir+'/cats-v-dogs')
# os.mkdir(datadir+'/cats-v-dogs/training')
# os.mkdir(datadir+'/cats-v-dogs/testing')
# os.mkdir(TESTING_CATS_DIR)
# os.mkdir(TRAINING_CATS_DIR)
# os.mkdir(TRAINING_DOGS_DIR)
# os.mkdir(TESTING_DOGS_DIR)

def split_data(SOURCE,TRAINING,TESTING,SPLIT_SIZE):
    filename=[]
    for fname in os.listdir(SOURCE):
        file=SOURCE+fname

        if os.path.getsize(file)>0:
            filename.append(fname)
    filename.sort()
    random.seed(230)
    random.shuffle(filename)
    training=int(SPLIT_SIZE*len(filename))
    training_img=filename[:training]
    testing_img=filename[training:]


    for file in training_img:
        sourcefilename= SOURCE+file
        destfilename=TRAINING+file

        #pdb.set_trace()
        shutil.copyfile(sourcefilename,destfilename)

    for file in testing_img:
        sourcefilename=SOURCE+file
        destfilename = TESTING + file
        shutil.copyfile(sourcefilename,destfilename)


#pdb.set_trace()
split_size = .9
#split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
#split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

def get_model():
    model =tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),activation=tf.nn.relu,input_shape=(150,150,3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
    return model

model=get_model()
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['acc'])

TRAINING_DIR=datadir+'/cats-v-dogs/training'
#train_datagen=ImageDataGenerator(rescale=1/255.0)
## Data Augmentation
train_datagen=ImageDataGenerator(rescale=1/255.0,
                                 rotation_range=60,
                                 featurewise_center=True,
                                 shear_range=0.2,samplewise_std_normalization=True,
                                 zoom_range=0.2,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2)

train_generator =train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                   batch_size=10,
                                                   target_size=(150,150),
                                                   class_mode='binary')

VALIDATION_DIR=datadir+'/cats-v-dogs/testing'
#validation_datagen=ImageDataGenerator(rescale=1/255.0)
validation_datagen=ImageDataGenerator(rescale=1/255.0,
                                 rotation_range=60,
                                 featurewise_center=True,
                                 shear_range=0.2,samplewise_std_normalization=True,
                                 zoom_range=0.2,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2)


validation_generator=validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                            target_size=(150,150),
                                                            class_mode='binary',
                                                            batch_size=10)

history =model.fit(train_generator,epochs=2,validation_data=validation_generator)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

plt.show()
model.load_weights()
