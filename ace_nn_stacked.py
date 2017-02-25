# Train this by running sbatch ace_rsdata_job.sh

######
# Simple Neural Net for use on the ace cluster
######
# A simple neural network that predicts depression diagnosis when given 16 raw slices of fMRI data


import sys

load_network = False
nb_layers_to_train = 1 if(len(sys.argv) <= 2) else int(sys.argv[2])
name = sys.argv[1] if (len(sys.argv) > 1) else 0

import os    
os.environ['THEANO_FLAGS'] = "device=gpu;floatX=float32"    

# Get the data
import numpy as np
from ace_rsdata import *

X, Y = get_dataset('DX')
input_shape = (16,96,96,50) # using new x preprocessing

nb_classes = 4

# Assemble the NN
model_file = 'h5_files/ace_rsdata_nn_stacked_16slices_{:}_02_25.h5'.format(name)
model = None

print("Training " + model_file)

# Create preprocessing network

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

print "Post preprocessing input shape: " + str(input_shape)

model = None
if load_network:
    model = keras.models.load_model(model_file)
else:
    model = Sequential()
    model.add(AveragePooling3D(pool_size=(3,3,3), input_shape=input_shape))    
    model.add(Flatten())
    model.add(Dense(nb_classes))


# Preprocess the data
print("Preprocessing the data ...")
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Decreasing input size to dramatically reduce training time for testing
# X, Y = X[0:7], Y[0:7]


print("Preprocessing: Loading and reshaping")

# Gets the 1st 16 timeslices and returns data in the shape (16, 96, 96, 50) (time, x, y, z)
def x_preprocess(x):
    return np.moveaxis(x.get_data(), -1, 0)[0:16]
# Gets the 1st timeslice
def old_x_preprocess(x):
    return np.expand_dims(np.moveaxis(x.get_data(), -1, 0)[0], axis=-1)

X = [ x_preprocess(x) for x in X]
Y = np_utils.to_categorical([y-1 for y in Y], nb_classes=nb_classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

Y_train = np.array(Y_train)
X_train_layer = np.array(X_train)
X_test_layer = np.array(X_test)
thin_model = model

def clone_model(model):
    new_model = Sequential.from_config(model.get_config())
    new_model.set_weights(model.get_weights())
    return new_model

# thin_model = keras.models.load_model(model_file)
for i in range(nb_layers_to_train):
    print("Training next layer when we have {:} layers in main model.".format(len(model.layers)))
    # Get the output of the last convolutional layer from the previous model:
    m = clone_model(thin_model)
    m.pop()
    m.pop()
    m.compile(loss='categorical_crossentropy',
        optimizer='adadelta')

    input_shape = m.layers[-1].output_shape[1:]
    X_train_layer = m.predict(X_train_layer)
    X_test_layer = m.predict(X_test_layer)

    # The new model we're going to train:
    thin_model = Sequential()
    layer = Convolution3D(16, 5, 5, 5,
             input_shape=input_shape,
             border_mode='same',
             activation='relu',
             name='c{:}'.format(len(model.layers)))
    thin_model.add(layer)
    thin_model.add(Flatten())
    thin_model.add(Dense(nb_classes))
    thin_model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    # Train the model
    thin_model.fit(X_train_layer, Y_train, nb_epoch=100, batch_size=10)
    
    loss = thin_model.evaluate(np.array(X_test_layer), Y_test)
    print('Loss with ', len(model.layers)+1 , ' layers: ', loss)
    
    # Save the combined model
    model.summary()
    model = clone_model(model)
    model.pop()
    model.pop()
    for layer in thin_model.layers:
        model.add(layer)

    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    model.save(model_file +  "_layers_{:}".format(len(model.layers)))

model.summary()
loss = model.evaluate(np.array(X_test), np.array(Y_test))
print(loss)
print("All done. :-)")

