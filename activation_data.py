
# coding: utf-8

# In[5]:

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, Deconvolution2D
from keras.utils import np_utils
from keras.backend.common import _FLOATX
import pprint
import pickle
import inspect
K.set_image_dim_ordering('th')

batch_size = 128
nb_classes = 10  # 10 digits from 0 to 9

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
training_data = X_train
Y_test = np_utils.to_categorical(y_test, nb_classes)

ex_model = load_model('example_model.h5')


# In[9]:


newlist = [0]

for index in range(0, len(ex_model.layers)):
    
    l = ex_model.layers[index]
    print(l.__class__)
    
    if isinstance(l, Activation):
        newlist.append(index)

for index in range(0, len(training_data[:400])):
    element = [[training_data[index]]]
    for index in range(len(newlist) - 1):
        f = K.function([ex_model.layers[newlist[index]].input, K.learning_phase()],
                       [ex_model.layers[newlist[index + 1]].output])
        output = f([element[0], 0])
        element = output
        
    with open('activation_data.pickle', 'wb') as handle:
      pickle.dump(output, handle)
