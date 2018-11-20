import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

(x_train,y_train), (x_test,y_test) =mnist.load_data() # Is it that easy to load the dataset?

img_rows,img_cols = 28,28 # What is the standart of the pixels? Do these images must have the same pixels? 299x299 224x224

batch_size = 128 # What should be the value of batch_size? RAM Usage?
num_classes = 10 # 0-9

epochs = 15 # How to determine the value of epoch number?



x_train = np.expand_dims(x_train,axis=1).reshape(-1,28,28,1)
x_test = np.expand_dims(x_test,axis=1).reshape(-1,28,28,1)


x_train = x_train/255
y_train = y_train/255
x_test = x_test/255
y_test = y_test/255

# What should be the format of scaling? Are there any other scaling methods?

print(y_train)

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
# One Hot encoding method.
#600000,28,28,1
#600000,26,26,1
#600000,24,24,1
#600000,12,12,1
#600000,12,12,1
#600000x12x12x1



model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation="softmax"))

print(model.summary())

# Finding qucick solutions to the choosing the right architecture for our dataset.
# Fine Tuning / Transfer Lerarning



model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=["accuracy"])


model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=[x_test,y_test])



# 10 classes, training 60000/10 - 6000 images per each class
# Imbalanced Dataset

# Image Augmentation - balances the data with creating artificial data. 100000 , 1000 + 9000