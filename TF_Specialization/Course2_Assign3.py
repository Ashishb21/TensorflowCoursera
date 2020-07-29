import tensorflow as tf
import pandas as pd
import os
import zipfile
import shutil

# download weights Inception V3
#wget https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
# -O Data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

#wget https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
# -O Data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

basepath='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
datadir =os.path.join(basepath,'Data')
weightpath=datadir+'/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(weightpath)

pre_trained_model.trainable=False
# for layer in pre_trained_model.layers:
#     layer.trainable = False

#print(pre_trained_model.summary())

last_layer=pre_trained_model.get_layer('mixed7')
print(last_layer.shape)