import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

base_dir='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir =base_dir+'Data/SVHN/'

train=loadmat(data_dir+'train_32x32.mat')
test=loadmat(data_dir+'test_32x32.mat')

train_images=train['X']
train_labels=train['y']
test_images=test['X']
test_labels=test['y']

print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)

train_images=np.transpose(train_images,((3,0,1,2)))
test_images=np.transpose(test_images,((3,0,1,2)))
print(train_images.shape)
print(test_images.shape)


def plot_images(train_data, train_label, cmap=None):
    fig = plt.figure(figsize=(15, 7))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        random_index = np.random.randint(0, train_data.shape[0])
        img = train_data[random_index, :, :]
        fig.add_subplot(rows, columns, i)
        random_label = train_label[random_index, :]
        plt.xlabel(f"{random_label}")
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
    plt.show()


plot_images(train_images, train_labels)

train_images_grayscale=tf.reduce_mean(train_images,axis=-1,keepdims=True)
test_images_grayscale=tf.reduce_mean(test_images,axis=-1,keepdims=True)
train_images_plotting=tf.reduce_mean(train_images,axis=-1,keepdims=False)

print(train_images_grayscale.shape)
print(test_images_grayscale.shape)
print(train_images_plotting.shape)

plot_images(train_images_plotting,train_labels,cmap='gray')

print(np.unique(train_labels))
for i,item in enumerate(train_labels):
    if item==10:
        train_labels[i]=0
for i,item in enumerate(test_labels):
    if item==10:
        test_labels[i]=0

print(np.unique(train_labels))
print(np.unique(test_labels))

train_label_onehot=tf.keras.utils.to_categorical(train_labels)
test_label_onehot=tf.keras.utils.to_categorical(test_labels)

def get_MLP(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=get_MLP(train_images_grayscale[0].shape)
print(model.summary())

def get_early_stopping():
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    return earlystopping

def get_checkpoint_every_epoch():
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_every_epoch/checkpoint_{epoch:002d}'
                                                         , save_freq='epoch', save_weights_only=True)
    return modelcheckpoint

def get_checkpoint_best_only():

    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_best_only/checkpoint',
                                                         monitor='val_accuracy', save_best_only=True,
                                                         mode='max', save_weights_only=True)
    return checkpoint_best

checkpoint_every_epoch = get_checkpoint_every_epoch()
checkpoint_best_only = get_checkpoint_best_only()
early_stopping = get_early_stopping()

callbacks = [checkpoint_every_epoch, checkpoint_best_only, early_stopping]
history=model.fit(train_images_grayscale, train_label_onehot, epochs=30, validation_split=0.15, callbacks=callbacks,batch_size=512)

# Plotting required curves
df = pd.DataFrame(history.history)

# Training set loss plot
loss_plot = df.plot(y = 'loss', title = 'Loss vs Epochs', legend = False)
loss_plot.set(xlabel = 'Epochs', ylabel = 'Loss')

# Training set accuracy plot
accuracy_plot = df.plot(y = 'accuracy', title = 'Categorical accuracy vs Epochs', legend = False)
accuracy_plot.set(xlabel = 'Epochs', ylabel = 'Accuracy')

# Validation set loss plot
val_loss_plot = df.plot(y = 'val_loss', title = 'Validation loss vs Epochs', legend = False)
val_loss_plot.set(xlabel = 'Epochs', ylabel = 'Validation loss')

# Validation set accuracy plot
val_acc_plot = df.plot(y = 'val_accuracy', title = 'Validation accuracy vs Epochs', legend = False)
val_acc_plot.set(xlabel = 'Epochs', ylabel = 'Validation accuracy')

results = model.evaluate(test_images_grayscale, test_label_onehot, batch_size = 32, verbose = 1)

print(f"Loss on the test set is as follows: {results[0]}")
print(f"Accuracy on the test set is as follows: {results[1]}")

def get_CNN(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                     input_shape=input_shape, name='conv_1'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_1'))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='conv_2'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='dense_2'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model=get_CNN(train_images_grayscale[0].shape)
print(model.summary())

def get_early_stopping():
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    return earlystopping

def get_checkpoint_every_epoch_cnn():
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_every_epoch_cnn/checkpoint_{epoch:002d}'
                                                         , save_freq='epoch', save_weights_only=True)
    return modelcheckpoint

def get_checkpoint_best_only_cnn():

    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints_best_only_cnn/checkpoint',
                                                         monitor='val_accuracy', save_best_only=True,
                                                         mode='max', save_weights_only=True)
    return checkpoint_best

checkpoint_every_epoch_cnn = get_checkpoint_every_epoch()
checkpoint_best_only_cnn = get_checkpoint_best_only()
early_stopping = get_early_stopping()

callbacks = [checkpoint_every_epoch_cnn, checkpoint_best_only_cnn, early_stopping]
history=model.fit(train_images_grayscale, train_label_onehot, epochs=30, validation_split=0.15, callbacks=callbacks,batch_size=512)

# Plotting required curves
df = pd.DataFrame(history.history)

# Training set loss plot
loss_plot = df.plot(y = 'loss', title = 'Loss vs Epochs', legend = False)
loss_plot.set(xlabel = 'Epochs', ylabel = 'Loss')

# Training set accuracy plot
accuracy_plot = df.plot(y = 'accuracy', title = 'Categorical accuracy vs Epochs', legend = False)
accuracy_plot.set(xlabel = 'Epochs', ylabel = 'Accuracy')

# Validation set loss plot
val_loss_plot = df.plot(y = 'val_loss', title = 'Validation loss vs Epochs', legend = False)
val_loss_plot.set(xlabel = 'Epochs', ylabel = 'Validation loss')

# Validation set accuracy plot
val_acc_plot = df.plot(y = 'val_accuracy', title = 'Validation accuracy vs Epochs', legend = False)
val_acc_plot.set(xlabel = 'Epochs', ylabel = 'Validation accuracy')


results = model.evaluate(test_images_grayscale, test_label_onehot, batch_size = 32, verbose = 1)
print(f"Loss on the test set is as follows: {results[0]}")
print(f"Accuracy on the test set is as follows: {results[1]}")

######## Get predictions ###################

model= get_MLP(input_shape=train_images_grayscale[0].shape)
model.load_weights('checkpoints_best_only/checkpoint')

# Plotting 5 random images and their respective plots
num_test_images = test_images_grayscale.shape[0]

random_inx = np.random.choice(num_test_images, 5)
random_test_images = test_images_grayscale[random_inx, ...]
random_test_labels = test_labels[random_inx, ...]

predictions = model.predict(random_test_images)

fig, axes = plt.subplots(5, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'Digit {label}')
    axes[i, 1].bar(np.linspace(1.0, 10.0, num=10), prediction)
    axes[i, 1].set_xticks(np.linspace(1.0, 10.0, num=10))
    axes[i, 1].set_title(f"Categorical distribution. Model prediction: {np.argmax(prediction) + 1}")
plt.show()


model= get_CNN(input_shape=train_images_grayscale[0].shape)
model.load_weights('checkpoints_best_only_cnn/checkpoint')




