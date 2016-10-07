## Multipurpose max patch functions
import numpy as np
from keras import backend as K
import keras

# Get a list of layers that can be used in max_patch
def get_convolutional_layers(model):
    legal_classes = (keras.layers.Convolution2D, keras.layers.convolutional.ZeroPadding2D)
    return [ layer for layer in model.layers if isinstance(layer, legal_classes)]


# Tested/built for Theano... should be tensorflow compatible
def max_patch(model, data, images=None, layer=None, layer_number=-1, filter_number=0, number_of_patches=9, patch_size=(8,8)):
    
    # images are unpreprocessed data
    if images == None:
        images = data
    
    # Layer is an optional argument
    if layer == None:
        layer = model.layers[layer_number]
    
    # Make sure the layer is a convolutional layer
    if not isinstance(layer, (keras.layers.Convolution2D, keras.layers.convolutional.ZeroPadding2D)):
        print('Hey! Your layer is of class {:}. Are you sure you want to get a 2D max patch of it?'.format(layer.__class__))
    
    # Has shape (1), where each element is the layer's output.
    # A typical layer's output is (1, filters, width, height)
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [layer.output])
    
    # List of np.arrays(shape=(width, height))
    outputs = [get_layer_output([inputs, 0])[0][0][filter_number] for inputs in data]
    
    # Get the maximum values
    maxes = [output.argmax() for output in outputs]
    
    # The indices of the images with the n highest maxes
    image_indices = np.argsort(maxes)[:number_of_patches]
    
    max_outputs = [ outputs[index] for index in image_indices]
    
    # Maximum locations in each 'image'
    # list of (x, y) locations... (technically, (x,y,z,q) locations are fine too)
    max_locations = [np.unravel_index(output.argmax(), output.shape) for output in max_outputs]

    
    # Works for multidimensional input
    # Get the location of the centers as fractions (between 0 and 1)
    # List of (index, (x,y)) where 0 < x < 1
    #fractional_centers = []    
    #for index in range(len(outputs)):
    #    fractions = [loc/total for loc, total in zip(max_locations[index], outputs[index].shape)]
    #    fractional_centers.append(tuple(fractions))    
    
    
    
    # Works only for 2D images
    def patch_from_location(image, max_location, patch_size):
        x = int(max_location[1]/outputs[0].shape[-1]*image.shape[1])
        y = int(max_location[0]/outputs[0].shape[-2]*image.shape[0])
        top = y-patch_size[0]//2
        left = x-patch_size[1]//2
        print(max_location, top, left)
        print(max_location,'*', image.shape, '/', outputs[0].shape)
        return image[top:top+patch_size[0],
                     left:left+patch_size[1]]
    
    patches = [patch_from_location(images[image_indices[index]], max_locations[index], patch_size)
            for index in range(len(image_indices))]
    
    return patches


# Use googlenet
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from googlenet.googlenet import *

    # GoogLeNet preprocesses images and messes with their dimensions, which makes it a pain to use matplotlib.
    # This allows us to create patches that are easy to plot
    import matplotlib.image as mpimg

    model = create_googlenet('googlenet_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    original_image = mpimg.imread("cat.jpg")

    img = imresize(imread('cat.jpg', mode='RGB'), (224, 224)).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)




    data = [img]*2 # input to the neural net
    images = [original_image]*2 # Used for generating patches - we cannot call plt.imshow(data[0])

    layers = get_convolutional_layers(model) # Get a list of the convolutional layers

    patches = max_patch(model, data, images, layer=layers[8], filter_number=34, patch_size=(40,50))

    for patch in patches:
        plt.figure()
        plt.imshow(patch)
        plt.show()