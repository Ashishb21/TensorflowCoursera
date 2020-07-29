import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt

base_dir='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir=base_dir+'Data/'

vocab_size=1000
samples=20000
max_len=120
padding='post'
trunc='post'
embedded_dim=16
num_epochs=10

url=[]
label=[]
sentence=[]

with open(data_dir+'sarcasm.json') as f:
    data=json.load(f)

for item in data :
    url.append(item['article_link'])
    sentence.append(item['headline'])
    label.append(item['is_sarcastic'])

print(sentence[0])
print(label[0])
print(np.unique(label))
print(len(sentence))
train_sentence=sentence[0:samples]
train_label=label[0:samples]
test_sentence=sentence[samples:]
test_label=label[samples:]

print(len(train_sentence))
print(len(train_label))
print(len(test_sentence))
print(len(test_label))

token=Tokenizer(oov_token='<OOV>',num_words=vocab_size)
token.fit_on_texts(train_sentence)

sequence=token.texts_to_sequences(train_sentence)
train_padded=pad_sequences(sequences=sequence,maxlen=max_len,padding=padding,truncating=trunc)

sequence=token.texts_to_sequences(test_sentence)
test_padded=pad_sequences(sequences=sequence,maxlen=max_len,padding=padding,truncating=trunc)

train_padded=np.array(train_padded)
test_padded=np.array(test_padded)
train_label=np.array(train_label)
test_label=np.array(test_label)


###### LSTM ##############
model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedded_dim,input_length=max_len))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tf.keras.layers.Dense(24,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))

################# LSTM with 1D Convolution #########

model_1=tf.keras.Sequential()
model_1.add(tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedded_dim,input_length=max_len))
model_1.add(tf.keras.layers.Conv1D(128,5,activation=tf.nn.relu))
model_1.add(tf.keras.layers.GlobalMaxPooling1D())
model_1.add(tf.keras.layers.Dense(24,activation=tf.nn.relu))
model_1.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))

#########################################

print(model.summary())
print(model_1.summary())

#model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
model_1.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])


#history=model.fit(x=train_padded,y=train_label,epochs=num_epochs,validation_data=(test_padded,test_label))
history=model_1.fit(x=train_padded,y=train_label,epochs=num_epochs,validation_data=(test_padded,test_label))
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()





