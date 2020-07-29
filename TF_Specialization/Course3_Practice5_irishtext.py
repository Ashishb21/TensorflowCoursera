import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

base_dir='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir=base_dir+'Data/'
# wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt \
#     -O /Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/Data/irish-lyrics-eof.txt

with open(data_dir+'irish-lyrics-eof.txt')as f:
    data=f.read()
#print(data)
corpus=data.lower().split('\n')
#print(corpus)

token=Tokenizer()
token.fit_on_texts(corpus)
total_word=token.word_index
total_word=len(total_word)+1
#print(total_word)

# sequence=token.texts_to_sequences(corpus)
# print(sequence)

input_data=[]
for line in corpus:
    sequence=token.texts_to_sequences([line])[0]
    for i in range(1,len(sequence)):
        se=sequence[0:i+1]
        input_data.append(se)
print(input_data)

max_length=max([len(item)for item in input_data])
print(max_length)

input_data_sequence=np.array(pad_sequences(sequences=input_data,maxlen=max_length,padding='pre',))
print(input_data_sequence[4])

xs=input_data_sequence[:,:-1]
xlabel=input_data_sequence[:,-1]
print(xs[4])
print(xlabel[4])
xl=tf.keras.utils.to_categorical(xlabel,num_classes=total_word)
print(xl[4])

model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=total_word,output_dim=120,input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150,return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(total_word/2,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
model.add(tf.keras.layers.Dense(total_word,activation=tf.nn.softmax))

print(model.summary())

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
history=model.fit(x=xs,y=xl,epochs=100)
pd.DataFrame(history.history).plot(figsize=(5,3))
plt.show()



