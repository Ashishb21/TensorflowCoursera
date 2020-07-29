import tensorflow as tf
import numpy as np

x=np.ones(shape=(100,10,2,2))  # 100 -> no of samples 10 -> batch size
dataset=tf.data.Dataset.from_tensor_slices(x)
print(dataset)
print(dataset.element_spec)

datgen=tf.keras.preprocessing.image.ImageDataGenerator()
datgen.flow_from_directory()
datgen.flow()
data=tf.data.Dataset.from_generator()

