import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#wget https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5 -O /Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/Data/MobileNetV2.h5

dir_path='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_path=dir_path+'Data'
weightpath=data_path+'/MobileNetV2.h5'

'''
The model should use the input_shape in the function argument to set the shape in the Input layer.
The first and second hidden layers should be Conv2D layers with 32 filters, 3x3 kernel size and ReLU activation.
The third hidden layer should be a MaxPooling2D layer with a 2x2 window size.
The fourth and fifth hidden layers should be Conv2D layers with 64 filters, 3x3 kernel size and ReLU activation.
The sixth hidden layer should be a MaxPooling2D layer with a 2x2 window size.
The seventh and eighth hidden layers should be Conv2D layers with 128 filters, 3x3 kernel size and ReLU activation.
The ninth hidden layer should be a MaxPooling2D layer with a 2x2 window size.
This should be followed by a Flatten layer, and a Dense layer with 128 units and ReLU activation
The final layer should be a Dense layer with a single neuron and sigmoid activation.
All of the Conv2D layers should use 'SAME' padding.

'''
def get_model(input_shape):
    input=tf.keras.Input(shape=input_shape)
    x=tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(input)
    x=tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(x)
    x=tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x=tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(128,activation='relu')(x)
    output=tf.keras.layers.Dense(1,activation='sigmoid')
    model=tf.keras.Model(inputs=input,outputs=output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    return model


model=tf.keras.applications.MobileNetV2()
model.save('/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/Data/MobileNetV2_1.h5')

base_model=tf.keras.models.load_model('/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/Data/MobileNetV2_1.h5')
#print(base_model.summary())
def remove_head(pretrained_model):
    output=pretrained_model.get_layer(name='global_average_pooling2d').output
    model=tf.keras.Model(inputs=pretrained_model.input,outputs=output)
    return model

feature_extractor = remove_head(base_model)
#print(feature_extractor.summary())


def add_new_classifier_head(feature_extractor_model):
    model=tf.keras.Sequential()
    model.add(feature_extractor_model)
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    return model
new_model = add_new_classifier_head(feature_extractor)
#print(new_model.summary())

def freeze_pretrained_weights(model):
    model.get_layer('model').trainable=False
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
frozen_new_model = freeze_pretrained_weights(new_model)
print(frozen_new_model.summary())


