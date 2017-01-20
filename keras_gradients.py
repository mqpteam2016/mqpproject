# How to get gradients from an arbitrary keras model:
#gradients = gradients_on_batch(model.model, X_train, y_train)

# Inspired by:
# https://github.com/fchollet/keras/issues/3080

from keras.engine.training import Model
from keras import backend as K

# Returns a backend theano/tensorflow function that gets gradients
def _make_gradient_function(self):

    if not hasattr(self, 'gradient_function'):
        self.gradient_function = None

    if self.gradient_function is None:
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs = self.inputs + self.targets + self.sample_weights + [K.learning_phase()]
        else:
            inputs = self.inputs + self.targets + self.sample_weights

        training_updates = self.optimizer.get_updates(self._collected_trainable_weights,
                                                      self.constraints,
                                                      self.total_loss)

        training_gradients = self.optimizer.get_gradients(
            self.total_loss,
            self._collected_trainable_weights,
        )
        outputs =  training_gradients # + [self.total_loss] + self.metrics_tensors

        updates = self.updates + training_updates

        # DOES NOT returns loss and metrics. Updates weights at each call.
        self.gradient_function = K.function(inputs,
                                         outputs,
                                         #updates=updates,
                                         **self._function_kwargs)
    return self.gradient_function


def gradients_on_batch(self, x, y, sample_weight=None, class_weight=None):
        x, y, sample_weights = self._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            class_weight=class_weight)
        if self.uses_learning_phase and not isinstance(K.learning_phase, int):
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        ######self._make_gradient_function()
        _make_gradient_function(self)
        outputs = self.gradient_function(ins)
        if len(outputs) == 1:
            return outputs[0]
        return outputs



#possibly: x=_make_gradient_function(model.model)
#or more usefully: gradients = gradients_on_batch(model.model, X_train, y_train)


# Here's an example way to get a gradient from a Neural Network:
if __name__ == "__main__":
    '''Trains a simple convnet on the MNIST dataset.
    Gets to 99.25% test accuracy after 12 epochs
    (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
    '''

    from __future__ import print_function
    import numpy as np
    np.random.seed(1337)  # for reproducibility

    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K

    batch_size = 128
    nb_classes = 10
    nb_epoch = 2

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    #################################
    # And now to get the gradients: #
    gradients = gradients_on_batch(model.model, X_test, Y_test)

    for i in range(len(gradients)):
        print(gradients[i].shape)
        # Shape on conv. layers are:
        # (nb_filters, input_filters, filter_x, filter_y...)
    # Prints out:
    # Dropout & flatten layers don't have gradients
    '''
    (32, 1, 3, 3)
    (32,)
    (32, 32, 3, 3)
    (32,)
    (4608, 128)
    (128,)
    (128, 10)
    (10,)
    '''
