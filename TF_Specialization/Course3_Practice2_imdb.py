import tensorflow_datasets as tfds
import numpy as np
import ipdb
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


imdb,info=tfds.load(name='imdb_reviews',with_info=True,as_supervised=True)

print(imdb.keys())

train_imdb,test_imdb=imdb['train'],imdb['test']

train_sentences=[]
train_label=[]
test_sentences=[]
test_label=[]

for d,l in train_imdb:
    train_sentences.append(str(d.numpy()))
    train_label.append(l.numpy())

for d,l in test_imdb:
    test_sentences.append(str(d.numpy()))
    test_label.append((l.numpy()))

train_label_final=np.array(train_label)
test_label_final=np.array(test_label)

token =Tokenizer(oov_token='<OOV>',num_words=10000)
token.fit_on_texts(train_sentences)
#print(token.word_index)

train_sequences=token.texts_to_sequences(train_sentences)
train_padded =pad_sequences(sequences=train_sequences,padding='post',maxlen=120)

token.fit_on_texts(test_sentences)
test_sequences=token.texts_to_sequences(test_sentences)
test_padded=pad_sequences(sequences=test_sequences,padding='post',maxlen=120)

model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=10000,output_dim=16,input_length=120))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(6,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(x=train_padded,y=train_label_final,epochs=10,validation_data=(test_padded,test_label_final))
