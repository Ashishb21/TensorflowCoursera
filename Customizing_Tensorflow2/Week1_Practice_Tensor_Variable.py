import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(4,)))
#print(model.summary())

#print(model.weights)

x=tf.Variable(['Ashish'],dtype=tf.string)
#print(x)
fl=tf.Variable([12.34],tf.float64)
#print(fl)
i=tf.Variable([12],tf.int32)
#print(i)
###############################
input=tf.keras.Input(shape=(32,1),name='input')
x=tf.keras.layers.Conv1D(16,5,name='conv1')(input)
x=tf.keras.layers.MaxPool1D(name='maxpool')(x)
x=tf.keras.layers.Flatten(name='flatten')(x)
output=tf.keras.layers.Dense(1,activation='sigmoid',name='output')(x)
model=tf.keras.Model(inputs=input,outputs=output)

#print(model.summary())
#print(model.layers[1].get_weights())
#print(model.weights)
#print(model.layers[1].weights())
#print(model.layers[1].bias)
#print(model.get_layer(name='conv1').input)
#print(model.get_layer(name='conv1').output)

############# taking new model from existing model ###############
output1=model.get_layer(name='flatten').output
model2=tf.keras.Model(inputs=input,outputs=output1)
#print(model2.summary())

#############################################adding output layer to existing model ########
output2=tf.keras.layers.Dense(1,activation='softmax',name='outputlayer')(model2.output)

model3=tf.keras.Model(inputs=model2.input,outputs=output2)
#model3.summary()
###########################VGG 16 Pretrained model #########################

from tensorflow.keras.applications import VGG16

vgg16_model=VGG16()
#print(vgg16_model.summary())
#print(vgg16_model.layers)
vgg_layers=vgg16_model.layers
vgg_input=vgg16_model.input
vgg16_model.trainable=False
#output_layers=[layer.output for layer in vgg_layers]


#################

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4,activation='relu',kernel_initializer='random_uniform',bias_initializer='ones',input_shape=(4,)))
model.add(tf.keras.layers.Dense(2,activation='relu',kernel_initializer='lecun_normal',bias_initializer='ones'))
model.add(tf.keras.layers.Dense(4,activation='softmax'))

print(model.summary())
print(model.weights)
print(model.layers)
weights_layers_1=[l.weights[0].numpy() for l in model.layers]
#print(weights_layers_1)
bias_layer=[l.bias.numpy() for l in model.layers]
print(bias_layer)
print(len(model.trainable_variables))
print(len(model.non_trainable_variables))