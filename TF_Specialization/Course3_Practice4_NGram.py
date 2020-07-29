import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_dir='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir=base_dir+'Data/'

data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

token=Tokenizer()
# data1=data.replace('\n','')
# data1=list(data.lower())

corpus=data.lower().split("\n")
#print(data)

token.fit_on_texts(corpus)
words=token.word_index
total_words=len(words)+1
#print(token.word_index)
#print(len(words))
n_grams=[]
for line in corpus:
    print([line])
    token_list=token.texts_to_sequences([line])[0]
    print(token_list)
    for i in range(1,len(token_list)):
        grams=token_list[0:i+1]
        n_grams.append(grams)
#print(n_grams)

# Pad sequence
max_length=max([len(x) for x in n_grams])
print(max_length)

input_sequence =np.array(pad_sequences(sequences=n_grams,maxlen=max_length,padding='pre'))
#print(input_sequence)

xs,xlabel=input_sequence[:,:-1],input_sequence[:,-1]
x_label=tf.keras.utils.to_categorical(xlabel,num_classes=total_words)
print(input_sequence[5])
print(xs[5])
print(x_label[5])

model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=total_words,output_dim=64,input_length=max_length-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)))
model.add(tf.keras.layers.Dense(total_words,activation=tf.nn.softmax))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
#history=model.fit(x=xs,y=x_label,epochs=500)
#pd.DataFrame(history.history).plot(figsize=(5,3))
#plt.show()

#model.save_weights(data_dir+'Ngram.h5')

prediction_data="laurence went to dublin"

sequence =token.texts_to_sequences([prediction_data])[0]
print(sequence)

token_list=pad_sequences(sequences=[sequence],padding='pre',maxlen=max_length)
print(token_list)


