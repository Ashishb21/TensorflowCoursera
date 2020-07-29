import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def get_new_model(input_shape):
    """
    The model should use the input_shape in the function argument to set the input size in the first layer.
    The first layer should be a Conv2D layer with 16 filters, a 3x3 kernel size, a ReLU activation function and 'SAME' padding. Name this layer 'conv_1'.
    The second layer should also be a Conv2D layer with 8 filters, a 3x3 kernel size, a ReLU activation function and 'SAME' padding. Name this layer 'conv_2'.
    The third layer should be a MaxPooling2D layer with a pooling window size of 8x8. Name this layer 'pool_1'.
    The fourth layer should be a Flatten layer, named 'flatten'.
    The fifth layer should be a Dense layer with 32 units, a ReLU activation. Name this layer 'dense_1'.
    The sixth and final layer should be a Dense layer with 10 units and softmax activation. Name this layer 'dense_2'.
    This function should build a Sequential model according to the above specification. Ensure the
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    Your function should also compile the model with the Adam optimiser, sparse categorical cross
    entropy loss function, and a single accuracy metric.
    """
    model =tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape,name='conv_1'))
    model.add(tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding='same',activation='relu',name='conv_2'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(8,8),name='pool_1'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32,activation='relu',name='dense_1'))
    model.add(tf.keras.layers.Dense(10,activation='softmax',name='dense_2'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model


def get_checkpoint_every_epoch():
    """
    This function should return a ModelCheckpoint object that:
    - saves the weights only at the end of every epoch
    - saves into a directory called 'checkpoints_every_epoch' inside the current working directory
    - generates filenames in that directory like 'checkpoint_XXX' where
      XXX is the epoch number formatted to have three digits, e.g. 001, 002, 003, etc.
    """
    modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_every_epoch/checkpoint_{epoch:002d}'
                                                       ,save_freq='epoch',save_weights_only=True)
    return modelcheckpoint

def get_checkpoint_best_only():
    """
    This function should return a ModelCheckpoint object that:
    - saves only the weights that generate the highest validation (testing) accuracy
    - saves into a directory called 'checkpoints_best_only' inside the current working directory
    - generates a file called 'checkpoints_best_only/checkpoint'
    """
    checkpoint_best=tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_best_only/checkpoint',
                                                       monitor='val_acc',save_best_only=True,
                                                       mode='max',save_weights_only=True)
    return checkpoint_best


def get_early_stopping():
    """
    This function should return an EarlyStopping callback that stops training when
    the validation (testing) accuracy has not improved in the last 3 epochs.
    HINT: use the EarlyStopping callback with the correct 'monitor' and 'patience'
    """''
    earlystopping=tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=3,mode='max')
    return earlystopping


def get_model_last_epoch(model):
    """
    This function should create a new instance of the CNN you created earlier,
    load on the weights from the last training epoch, and return this model.
    """
    path = os.path.join(os.getcwd(), 'checkpoints_every_epoch')
    weightpath = tf.train.latest_checkpoint('checkpoints_every_epoch')
    model.load_weights(path)
    return model


def get_model_best_epoch(model):
    """
    This function should create a new instance of the CNN you created earlier, load
    on the weights leading to the highest validation accuracy, and return this model.
    """
    path = os.path.join(os.getcwd(), 'checkpoints_best_only')
    weightpath = tf.train.latest_checkpoint(path)
    model
    return model


def get_model_eurosatnet():
    """
    This function should return the pretrained EuroSatNet.h5 model.
    Cointains full model and weights 
    """
    model=tf.keras.models.load_model('models/EuroSatNet.h5')
    return model



