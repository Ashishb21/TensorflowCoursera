import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir ='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir=base_dir+'Data/'

data_set=pd.read_csv(data_dir+'diagnosis.csv')
print(data_set.head())
data=data_set.values

x_train,x_test,y_train,y_test=train_test_split(data[:,0:6],data[:,6:],test_size=0.2)

x_train_1,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6=np.transpose(x_train)
x_test_1,x_test_2,x_test_3,x_test_4,x_test_5,x_test_6=np.transpose(x_test)
y_train_1,y_train_2=y_train[:,0],y_train[:,1]
y_test_1,y_test_2=y_test[:,0],y_test[:,1]


print(x_train_1.shape)
print(y_train_1.shape)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


f1=tf.keras.Input(shape=(1,),name='f1')
f2=tf.keras.Input(shape=(1,),name='f2')
f3=tf.keras.Input(shape=(1,),name='f3')
f4=tf.keras.Input(shape=(1,),name='f4')
f5=tf.keras.Input(shape=(1,),name='f5')
f6=tf.keras.Input(shape=(1,),name='f6')
concat=tf.keras.layers.concatenate([f1,f2,f3,f4,f5,f6])
output1=tf.keras.layers.Dense(1,activation='sigmoid',name='output1')(concat)
output2=tf.keras.layers.Dense(1,activation='sigmoid',name='output2')(concat)


model=tf.keras.Model(inputs=[f1,f2,f3,f4,f5,f6],outputs=[output1,output2])
print(model.summary())
#tf.keras.utils.plot_model(model,to_file='week1_prac_model.png')

model.compile(optimizer='adam',loss=['binary_crossentropy','binary_crossentropy'],loss_weights=[1,2],metrics=['accuracy'])
input_train=[x_train_1,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6]
output_train=[y_train_1,y_train_2]

history=model.fit(x=input_train,y=output_train,epochs=1000,batch_size=128)
pd.DataFrame(history.history).plot(figsize=(5,3))
plt.show()
