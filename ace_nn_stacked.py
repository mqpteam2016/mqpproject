# Train this by running sbatch ace_rsdata_job.sh

######
# Simple Neural Net for use on the ace cluster
######
# A simple neural network that predicts depression diagnosis when given 16 raw slices of fMRI data
from __future__ import print_function

import sys

load_network = False
nb_layers_to_train = 1 if(len(sys.argv) <= 2) else int(sys.argv[2])
name = sys.argv[1] if (len(sys.argv) > 1) else 0
epochs = 25
quick_and_dirty_test = False

import os
os.environ['THEANO_FLAGS'] = "device=gpu;floatX=float32"

# Get the data
import numpy as np
from ace_rsdata import *

X, Y = get_dataset('DX')
input_shape = (16,96,96,50)

nb_classes = 4

# Assemble the NN
model_file = 'h5_files/ace_rsdata_nn_stacked_16slices_{:}_03_21.h5'.format(name)
model = None

print("Training " + model_file)


# Dumb functions because h5py isn't working right now...
import pickle
def save(model, filename):
    """Hacky saving of a sequential model. Does not save optimizer"""
    data = {
        'json': model.to_json(),
        'weights': [layer.get_weights() for layer in model.layers]
    }
    pickle.dump(data, open(filename, "wb" ))

def load(filename, custom_objects=None):
    """Hacky loading of a model. Does not load optimizer or compile model"""
    from keras.models import model_from_json
    data=pickle.load(open(filename, 'rb'))
    model = model_from_json(data['json'], custom_objects)

    [layer.set_weights(data['weights'][index]) for index, layer in enumerate(model.layers)]

    return model



# Create preprocessing network

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

print("Post preprocessing input shape: " + str(input_shape))

model = None
if load_network:
    model = keras.models.load_model(model_file)
else:
    model = Sequential()
    model.add(AveragePooling3D(
        pool_size=(1,3,3),
        input_shape=input_shape))
    model.add(Dropout(0))
    model.add(Flatten())
    model.add(Dense(nb_classes))

# Preprocess the data
print("Preprocessing the data ...")
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

# Decreasing input size to dramatically reduce training time for testing
if quick_and_dirty_test:
    epochs = 1
    X, Y = X[0:7], Y[0:7]


print("Preprocessing: Loading and reshaping")

# Gets the 1st 16 timeslices and returns data in the shape (16, 96, 96, 50) (time, x, y, z)
def x_preprocess(x):
    return np.moveaxis(x.get_data(), -1, 0)[0:16]
    # return x.get_data()[:,:,:,0:16] # gives us shape (96, 96, 50, 16), which I think is wrong

X = [ x_preprocess(x) for x in X]
Y = np_utils.to_categorical([y-1 for y in Y], nb_classes=nb_classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Get the output of a particular layer...
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input], layer.output)
    activations = get_activations([X_batch])
    return activations

x=0
def conv(weights=None):
    global x
    x=x+1
    if weights == []:
        return Dropout(0, input_shape=input_shape)
    return Convolution3D(16, 5, 5, 5,
             input_shape=input_shape,
             border_mode='same',
             activation='relu',
             name='c{:}'.format(x),
             weights=weights)

Y_train = np.array(Y_train)
X_train_layer = get_activations(model, model.layers[-3], np.array(X_train))
X_test_layer = get_activations(model, model.layers[-3], np.array(X_test))

# thin_model = keras.models.load_model(model_file)
for i in range(nb_layers_to_train):
    print("Training last 2 layers of a network")
    print("Training next layer when we have {:} layers in main model.".format(len(model.layers)))

    # Setup new model
    input_shape = X_train_layer[0].shape
    thin_model = Sequential()
    thin_model.add(conv(model.layers[-3].get_weights()))
    thin_model.add(conv())
    thin_model.add(Flatten())
    thin_model.add(Dense(nb_classes))
    thin_model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    # Train the model
    thin_model.fit(X_train_layer, Y_train, nb_epoch=epochs, batch_size=10)

    # Let's evaluate the new layer(s):
    loss = thin_model.evaluate(np.array(X_test_layer), Y_test)
    print('Loss with ', len(model.layers)+1 , ' layers: ', loss)

    print("Saving...")
    # Than save that layer (and the ones that come after) onto the main model
    model.pop() # Pop dense layer
    model.pop() # Pop flatten layer
    model.pop() # Pop poorly trained final layer
    model.add(conv(thin_model.layers[-4].get_weights()))
    model.add(conv(thin_model.layers[-3].get_weights()))
    model.add(Flatten())
    model.add(Dense(nb_classes, weights=thin_model.layers[-1].get_weights()))
    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    #model.save(model_file +  "_layers_{:}".format(len(model.layers)))
    save(model, model_file +  "_layers_{:}".format(len(model.layers)))

    print("Preparing for training next layer:")
    # Find activations from the oldest currently training layer
    X_train_layer = get_activations(thin_model, thin_model.layers[0], X_train_layer)
    X_test_layer = get_activations(thin_model, thin_model.layers[0], X_test_layer)

model.summary()
loss = model.evaluate(np.array(X_test), np.array(Y_test))
print("Complete model loss:", loss)
print("All done. :-)")
