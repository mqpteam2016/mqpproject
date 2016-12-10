import numpy as np
from keras import backend as K
import keras

# A list of classes that are considered convolutional layers.
# Used to determine what layers are condidered convolutional in get_convolutional_layers
convolutional_classes = (
    keras.layers.Convolution2D,
    keras.layers.Convolution3D,
    keras.layers.convolutional.ZeroPadding2D,
    keras.layers.convolutional.ZeroPadding3D
)


"""A visualization for a particular filter for a Convolutional layer of a neural net.
Displays locations (and the area around them) on the input images/data
that causes the maximum(s) of activation of that filter.

This particular implementation picks at most one point from each picture.
"""
class MaxPatch:
    def __init__(self, model, data, images=None, layer=None, layer_number=-1, filter_number=0, number_of_patches=9, patch_size=(8,8)):
        self.model = model
        self.images = images if images != None else data
        self.layer = layer if layer != None else model.layers[layer_number]
 	self.filter_number  = filter_number
        self.data = data
	self.patch_size = patch_size
	self.number_of_patches = number_of_patches
	# Make sure the layer is a convolutional layer
        if not isinstance(self.layer, convolutional_classes):
                print('Hey! Your layer is of class {:}. Are you sure you want to get a max patch of it?'.format(self.layer.__class__))

        if (self.images[0].shape[-1] > self.layer.output_shape[-1] * patch_size[-1] or
                self.images[0].shape[-2] > self.layer.output_shape[-2] * patch_size[-2]):
            print('Hey! Your patch size is small for this layer\'s output relative to the original image. You might want to increase it.')

        # Has shape (1), where each element is the layer's output.
        # A typical layer's output is (1, filters, width, height)
        self.get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                            [self.layer.output])

	if len(self.patch_size) != len(self.images[0].shape):
	    print('Hey! Your patch size has the wrong number of dimensions ({:}) to match to an image ({:})!'.format(len(self.patch_size), len(self.images[0].shape)))

	if model.layers[0].input_shape[1:] != data[0].shape:
	    print('Data shape {:} does not match model input shape {:}.'.format(data[0].shape, model.layers[0].input_shape))


    def generate(self):
        """Does the heavy lifting to generate a MaxPatch visualization"""
        # List of np.arrays(shape=(width, height))
	self.outputs = [self.get_layer_output([np.array([inputs]), 0])[0][0][self.filter_number] for inputs in self.data]
        # Get the maximum values
        maxes = [output.argmax() for output in self.outputs]
        # The indices of the images with the n highest maxes
        image_indices = np.argsort(maxes)[:self.number_of_patches]
        max_outputs = [ self.outputs[index] for index in image_indices]

        # Maximum locations in each 'image'
        # list of (x, y) locations... (technically, (x,y,z,q) locations are fine too)
        max_locations = [np.unravel_index(output.argmax(), output.shape) for output in max_outputs]

        self.patches = [MaxPatch.patch_from_location(self.images[image_indices[index]], max_locations[index], self.patch_size, self.outputs)
                for index in range(len(image_indices))]

    def show(self):
        """Actually shows the visualization"""
        for patch in self.patches:
            plt.figure()
            plt.imshow(patch)
            plt.show()

    @staticmethod
    def get_convolutional_layers(model):
        """Get a list of layers that can be used in max_patch"""
        return [ layer for layer in model.layers if isinstance(layer, convolutional_classes)]

    @staticmethod
    def patch_from_location(image, max_location, patch_size, outputs):
        # Multidimensional way of getting a patch from a location
        ratios = np.array(image.shape)[-len(outputs[0].shape):].astype('float') / np.array(outputs[0].shape)
        image_max_location = ratios * max_location
        extents = []
        patch = image
        for i in range(len(ratios)):
            min_patch_index = image_max_location[i]-patch_size[i]//2
            min_patch_index = int(np.clip(min_patch_index, 0, image.shape[i]-patch_size[i]))
            patch = np.moveaxis(np.moveaxis(patch, i, 0)[min_patch_index:min_patch_index+patch_size[i]], 0, i)
        return patch

    @staticmethod
    def patch_from_location2D(image, max_location, patch_size, outputs):
        """Works only for 2D images"""  
        x_ratio = image.shape[1]/outputs[0].shape[-1]
        y_ratio = image.shape[0]/outputs[0].shape[-2]
        x = int(max_location[1] * x_ratio)
        y = int(max_location[0] * y_ratio)
        top = np.clip(y-patch_size[0]//2,    0, image.shape[0])
        left = np.clip(x-patch_size[1]//2,   0, image.shape[1])
        return image[top:np.clip(top+patch_size[0], 0, image.shape[0]),
            left:np.clip(left+patch_size[1], 0, image.shape[1])]





# Use googlenet
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..googlenet.googlenet import *

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

