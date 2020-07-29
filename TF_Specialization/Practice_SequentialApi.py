import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Building Complex network using sequential Api

input_layer=tf.keras.layers.Input(shape=[32,32,3])
hidden1=tf.keras.layers.Dense(64,activation=tf.nn.relu)(input_layer)
hidden2=tf.keras.layers.Dense(64,activation=tf.nn.relu)(hidden1)
concat=tf.keras.layers.Concatenate()([input_layer,hidden2])
output=tf.keras.layers.Dense(1,activation=tf.nn.relu)(concat)
model=tf.keras.Model(inputs=[input_layer],outputs=[output])


#print(model.summary())

input_a=tf.keras.Input(shape=(32,32,3))
input_b=tf.keras.Input(shape=(32,32,3))
hidden1=tf.keras.layers.Dense(32,activation=tf.nn.relu)(input_a)
hidden2=tf.keras.layers.Dense(32,activation=tf.nn.relu)(hidden1)
concat=tf.keras.layers.Concatenate()([hidden2,input_a])
output=tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(concat)
model=tf.keras.Model(inputs=[input_a,input_b],outputs=[output])

#print(model.summary())

# class Mymodel(tf.keras.Model):
#     def __init__(self):
#         super(Mymodel,self).__init__()
#         self.hidden1=tf.keras.layers.Dense(32,activation=tf.nn.relu)
#         self.output=tf.keras.layers.Dense(1,activation=tf.nn.softmax)
#
#     def call(self,inputs):

sentences=['I love dogs','I love cats','I love donkey','my name is beautiful house']

token=Tokenizer()
token.fit_on_texts(sentences)
x=token.word_index
y=token.texts_to_sequences(sentences)
print(x)
print(y)

##################################################

token=Tokenizer()
token.fit_on_texts(sentences)

sequences =token.texts_to_sequences(sentences)
padded =pad_sequences(sequences=sequences)
print(padded)
#################################################
