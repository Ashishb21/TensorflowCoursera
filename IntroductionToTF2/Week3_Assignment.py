import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets , model_selection


def read_in_and_split_data(iris_data):
    """
    This function takes the Iris dataset as loaded by sklearn.datasets.load_iris(), and then
    splits so that the training set includes 90% of the full dataset, with the test set
    making up the remaining 10%.
    Your function should return a tuple (train_data, test_data, train_targets, test_targets)
    of appropriately split training and test data and targets.

    If you would like to import any further packages to aid you in this task, please do so in the
    Package Imports cell above.
    """
    dataset =datasets.load_iris()
    train_data,test_data,train_targets,test_targets=model_selection.train_test_split(dataset.data,dataset.target,test_size=0.1)
    return train_data,test_data,train_targets,test_targets

iris_data = datasets.load_iris()
train_data, test_data, train_targets, test_targets = read_in_and_split_data(iris_data)

train_targets = tf.keras.utils.to_categorical(np.array(train_targets))
test_targets = tf.keras.utils.to_categorical(np.array(test_targets))

def get_model(input_shape):
    """
    The model should use the input_shape in the function argument to set the input size in the first layer.
    The first layer should be a dense layer with 64 units.
    The weights of the first layer should be initialised with the He uniform initializer.
    The biases of the first layer should be all initially equal to one.
    There should then be a further four dense layers, each with 128 units.
    This should be followed with four dense layers, each with 64 units.
    All of these Dense layers should use the ReLU activation function.
    The output Dense layer should have 3 units and the softmax activation function.
    """
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(),bias_initializer='ones',input_shape=input_shape))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

model = get_model(train_data[0].shape)
print(model.summary())

def compile_model(model):
    """
    This function takes in the model returned from your get_model function, and compiles it with an optimiser,
    loss function and metric.
    Compile the model using the Adam optimiser (with learning rate set to 0.0001),
    the categorical crossentropy loss function and accuracy as the only metric.
    Your function doesn't need to return anything; the model will be compiled in-place.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

compile_model(model)

def train_model( model,train_data, train_targets, epochs):
    """
    This function should train the model for the given number of epochs on the
    train_data and train_targets.
    Your function should return the training history, as returned by model.fit.
    Run the training for a fixed number of epochs, given by the function's epochs argument.
    Return the training history to be used for plotting the learning curves.
    Set the batch size to 40.
    Set the validation set to be 15% of the training set.
    """
    history=model.fit(x=train_data,y=train_targets,epochs=epochs,validation_split=0.15,batch_size=40)
    return history

def get_regularised_model(input_shape, dropout_rate, weight_decay):
    """
    This function should build a regularised Sequential model according to the above specification.
    The dropout_rate argument in the function should be used to set the Dropout rate for all Dropout layers.
    L2 kernel regularisation (weight decay) should be added using the weight_decay argument to
    set the weight decay coefficient in all Dense layers that use L2 regularisation.
    Ensure the weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument input_shape.
    Your function should return the model.
    Add a dropout layer after the 3rd Dense layer
    Then there should be two more Dense layers with 128 units before a batch normalisation layer
    Following this, two more Dense layers with 64 units and then another Dropout layer
    Two more Dense layers with 64 units and then the final 3-way softmax layer
    Add weight decay (l2 kernel regularisation) in all Dense layers except the final softmax layer
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(),
                                    bias_initializer='ones', input_shape=input_shape,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

def get_callbacks():
    """
        This function should create and return a tuple (early_stopping, learning_rate_reduction) callbacks.
        The callbacks should be instantiated according to the above requirements.
        The early stopping callback is used and monitors validation loss with the mode set to "min" and patience of 30.
        The learning rate reduction on plateaux is used with a learning rate factor of 0.2 and a patience of 20.
        """
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=30)
    learning_rate_reduction=tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,patience=20)
    return early_stopping,learning_rate_reduction



