# Train this by running sbatch ace_rsdata_job.sh

#!/work/mgaskell/mqpproject/ENV/bin/python
#SBATCH -N1
#SBATCH --partition=shared
#SBATCH --gres=gpu:2

######
# Simple Neural Net for use on the ace cluster
######
# A simple neural network that predicts Age when given a raw slice of MRI data

# The 'new' networks increase the number of featuremap dimensions in a way not unlike what the following paper recommends
# https://arxiv.org/pdf/1610.02915v2.pdf


train_network = True
variant = 2 

# Get the data
import numpy as np
from ace_rsdata import *
import os

X, Y = get_dataset('DX')
dummy_x = np.zeros(X[0].shape)
input_shape = np.expand_dims(np.moveaxis(dummy_x, -1, 0)[0], axis=-1).shape
print "Input shape: " + str(input_shape)

# Assemble the NN
model_file = 'h5_files/ace_rsdata_nn_variant{:}_02_10.h5'.format(variant)
model = None

try:
    import keras
    model = keras.models.load_model(model_file)
except IOError:
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Reshape
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, AveragePooling3D
    from keras.utils import np_utils
    from keras import backend as K
    K.set_image_dim_ordering('tf')

    if variant == 1:
        model = Sequential()
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(10, 10, 5)))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(5, 5, 3)))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
    elif variant == 2:
        model = Sequential()

        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(10, 10, 5)))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(5, 5, 3)))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))

    elif variant == 3:
        model = Sequential()

        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(10, 10, 5)))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(5, 5, 3)))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(24, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(48, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))

    elif variant == '1_old': # 5 -> 3 -> 1 layers
        model = Sequential()

        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(10, 10, 5)))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(5, 5, 3)))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
    elif variant == '2_old': # 1 -> 3 -> 5
        model = Sequential()

        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(10, 10, 5)))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(5, 5, 3)))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
    elif variant == '3_old': # 1 -> 3 -> 5, only one maxpooling layer
        model = Sequential()

        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        # model.add(MaxPooling3D(pool_size=(10, 10, 5)))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(20, 20, 15)))
        model.add(Convolution3D(16, 1, 1, 1, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 3, 3, 3, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution3D(16, 5, 5, 5, input_shape=input_shape, border_mode='same'))
        model.add(Activation('relu'))

    shape = model.layers[-1].output_shape
    model.add(Flatten())
    #model.add(Reshape((shape[1]*shape[2]*shape[3]*shape[4],)))
    model.add(Dense(4))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')


# Preprocess the data
print("Preprocessing the data ...")
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Decreasing input size to dramatically reduce training time for testing
# X, Y = X[0:7], Y[0:7]

Y = np_utils.to_categorical([y-1 for y in Y], nb_classes=4)

# X's shape is  ( , 96, 96, 50, 1) or (scans, x, y, z, dummy value)
X = [np.expand_dims(np.moveaxis(x.get_data(), -1, 0)[0], axis=-1) for x in X]
# X = [np.expand_dims(x.get_data(), axis=0) for x in X]

#Y = list(map(lambda x: x.values[0], Y))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train the NN and periodically save it
if train_network:
    print("Starting training")
    for i in range(0, 1000, 50):
        model.fit(np.array(X_train), np.array(Y_train), nb_epoch=50)
        print('Trained epoch ' + str(i))
        if i % 100 == 0:
            try:
                os.rename(model_file, model_file + '_epoch_' + str(i-100))
            except OSError, e:
                print(model_file + " Doesn't yet exist")
            model.save(model_file)

model.summary()
print(np.array(X_test).shape)
loss = model.evaluate(np.array(X_test), np.array(Y_test))
print(loss)
print("All done. :-)")
