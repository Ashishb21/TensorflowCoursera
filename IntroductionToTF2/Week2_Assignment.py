import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mnist_data=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist_data.load_data()


def scale_mnist_data(train_images,test_images):
    train_images=train_images/255.0
    test_images=test_images/255.0
    return train_images,test_images

scaled_train_images,scaled_test_images=scale_mnist_data(train_images,test_images)

scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

def get_model(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=8,activation='relu',padding='same',kernel_size=(3,3),input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    return model

model = get_model(scaled_train_images[0].shape)
print(model.summary())


def compile_model(model):
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

compile_model(model)


def train_model(model,scaled_train_images,train_labels,):
    history=model.fit(x=scaled_train_images,y=train_labels,epochs=5,)
    return history

history = train_model(model, scaled_train_images, train_labels)

frame = pd.DataFrame(history.history)
acc_plot = frame.plot(y="accuracy", title="Accuracy vs Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Accuracy")
acc_plot = frame.plot(y="loss", title = "Loss vs Epochs",legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Loss")

def evaluate_model( model,scaled_test_images, test_labels):
    test_loss,test_accuracy=model.evaluate(scaled_test_images,test_labels)
    return test_loss,test_accuracy

