import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense

# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt \
#     -O /Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/Data/sonnets.txt

base_dir='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir=base_dir+'Data/'

with open(data_dir+'sonnets.txt') as f:
    data=f.read()

corpus=data.lower().split("\n")

token=Tokenizer(oov_token='<OOV>')
token.fit_on_texts(corpus)
total_words=len(token.word_index)+1
print(total_words)

input_data=[]
for line in corpus:
    sequence=token.texts_to_sequences([line])[0]
    #print(sequence)
    for i in range(1,len(sequence)):
        seq=sequence[:i+1]
        input_data.append(seq)
#print(input_data)

max_length=max([len(x) for x in input_data])

train_data=np.array(pad_sequences(sequences=input_data,maxlen=max_length,padding='pre'))
print(train_data[4])

train_x=train_data[:,:-1]
train_y=train_data[:,-1]
train_y=tf.keras.utils.to_categorical(train_y)

model=tf.keras.Sequential()
model.add(Embedding(input_dim=total_words,output_dim=100,input_length=max_length-1))
model.add(Bidirectional(LSTM(150,return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
model.add(Dense(total_words,activation=tf.nn.softmax))

print(model.summary())
          
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
history=model.fit(x=train_x,y=train_y,epochs=10)
pd.DataFrame(history.history).plot(figsize=(5,3))
plt.show()






