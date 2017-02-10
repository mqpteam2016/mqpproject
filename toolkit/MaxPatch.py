import numpy as np
from keras import backend as K
import keras
from matplotlib import pyplot as plt

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
    def __init__(self, model, data, images=None, layer=None, layer_number=-1, filter_number=0, number_of_patches=9, patch_size=None):
        self.model = model
        self.images = images if images != None else data
        self.layer = layer if layer != None else model.layers[layer_number]
        self.filter_number = filter_number
        self.data = data
        self.patches = None
        if patch_size == None:
            patch_size = MaxPatch.calculate_patch_size(self.model, self.layer)
            print("Auto calculated patch size of {:}".format(patch_size))
        self.patch_size = patch_size
        self.number_of_patches = number_of_patches
        # Make sure the layer is a convolutional layer
        if not isinstance(self.layer, convolutional_classes):
                print('Hey! Your layer is of class {:}. Are you sure you want to get a max patch of it?'.format(self.layer.__class__))

        if (self.images[0].shape[-1] > self.layer.output_shape[-1] * self.patch_size[-1] or
                self.images[0].shape[-2] > self.layer.output_shape[-2] * self.patch_size[-2]):
            print('Hey! Your patch size is small for this layer\'s output relative to the original image. You might want to increase it.')

        # Has shape (1), where each element is the layer's output.
        # A typical layer's output is (1, filters, width, height)
        self.get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                            [self.layer.output])

        if len(self.patch_size)+1 != len(self.images[0].shape) and len(self.patch_size) != len(self.images[0].shape):
            print('Hey! Your patch size has the wrong number of dimensions ({:}) to match to an image ({:})! (patch size should have the same or one less dimension than image shape)'.format(len(self.patch_size), len(self.images[0].shape)))

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
        self.max_locations = []
        self.patches = [MaxPatch.patch_from_location(self.images[image_indices[index]], max_locations[index], self.patch_size, self.outputs, image_max_locations=self.image_max_locations)
                for index in range(len(image_indices))]
        
        self._uncorrected_max_locations = max_locations
        self.image_indices = image_indices

    def save(self, filename='patches.png', dimensions=2):
        """Saves a png of the patches into the specified file"""
        self.show(filename=filename, dimensions=dimensions)

    def show(self, dimensions=2, filename=None):
        """
        Actually shows the visualization.
        Assumes that patches are 2D images (with the 1st dimension as color)
        or 3D images (and dimensions=3)
        """
        fig = None
        if not dimensions == 3:
            fig = self._show2D()
        else:
            fig = self._show3D()

        if filename:
            fig.savefig(filename, bbox_inches='tight')
        else:
            fig.show()

    def _show3D(self):
        """Creates a matplotlib figure with different 'images' as columns, and their cross sections as rows."""
        if not self.patches:
            self.generate()

        fig, axarr = plt.subplots(self.patches[0].shape[-1], len(self.patches))

        fig.set_size_inches(len(self.patches), self.patches[0].shape[-1])
        
        fig.suptitle("Patches corresponding to maximally active locations on layer: {:}, filter: {:}".format(self.layer.name, self.filter_number), y=0.0)
        for i in range(len(self.patches)):
            # Reshaping patch
            patch = self.patches[i]
            if len(patch.shape) == 4:
                patch = np.squeeze(patch, axis=(0,))
            if not len(patch.shape) == 3:
                raise ValueError("Expected patches to be 3D (or 4D with the first dimension of 1). Got shape {:}".format(self.patches[i].shape))

            # For each patch, draw all cross sections
            for z in range(patch.shape[-1]):
                subplot = axarr[z,i]
                subplot.axis('off')
                subplot.set_title('Z={:}'.format(z))
                subplot.imshow(patch[:, :, z], cmap="gray", interpolation='none')
        return fig

    def _show2D(self):
        """Create a matplotlib figure to show"""
        if not self.patches:
            self.generate()

        fig, axarr = plt.subplots(1, len(self.patches))

        fig.suptitle("Patches corresponding to maximally active locations on layer: {:}, filter: {:}".format(self.layer.name, self.filter_number), y=0.4)
        
        for i in range(len(self.patches)):

            reshaped_patch = np.moveaxis(self.patches[i], 0, -1)

            if not reshaped_patch.shape[-1] in [1, 3, 4] and len(reshaped_patch.shape) in [2, 3]:
                raise ValueError("Expected patches to be 2D, and possibly with the 1st dimension as a color. Got shape {:}".format(self.patches[i].shape))
            if reshaped_patch.shape[-1] == 1:
                reshaped_patch = np.squeeze(reshaped_patch, -1)
            axarr[i].axis('off')
            axarr[i].imshow(reshaped_patch, cmap="gray", interpolation='none')
        return fig

    @staticmethod
    def get_convolutional_layers(model):
        """Get a list of layers that can be used in max_patch"""
        return [ layer for layer in model.layers if isinstance(layer, convolutional_classes)]

    @staticmethod
    def patch_from_location(image, max_location, patch_size, outputs, image_max_locations=[]):
        # Multidimensional way of getting a patch from a location

        # For a image of (1, 28, 28) and a filter output of (14, 14), this should produce (2, 2)
        ratios = np.array(image.shape)[-len(outputs[0].shape):].astype('float') / np.array(outputs[0].shape)
        image_max_location = ratios * max_location
        image_max_locations.append(image_max_location)
        extents = []
        patch = image
        # Clip the last len(outputs[0].shape) to the patch size (e.g. for an image of (3, 28, 28), make it (3, 8, 8))
        for axis in range(-1,-(len(ratios)+1), -1): # Go through the axis, starting at Z
            min_patch_index = image_max_location[axis]-patch_size[axis]//2
            min_patch_index = int(np.clip(min_patch_index, 0, image.shape[axis]-patch_size[axis]))
            patch = np.moveaxis(np.moveaxis(patch, axis, 0)[min_patch_index:min_patch_index+patch_size[axis]], 0, axis)
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

    # The shape of the section of the input to a layer that fully determines a pixel of the layer's output
    @staticmethod
    def _input_multiplier(layer):
        if hasattr(layer, 'pool_size'):
            return layer.pool_size # For pooling layers
        elif hasattr(layer, 'filter_length'):
            return (layer.filter_length,) # For 1D convolutional layers
        elif hasattr(layer, 'nb_row') and hasattr(layer, 'nb_col'):
            return (layer.nb_row, layer.nb_col)
        elif (hasattr(layer, 'kernel_dim1')
          and hasattr(layer, 'kernel_dim2')
          and hasattr(layer, 'kernel_dim3')):
            # 3D convolutional layer
            return (layer.kernel_dim1, layer.kernel_dim2, layer.kernel_dim3)
        else:
            return (1, 1,1,1) # Assume it's an activation layer or the like

        # The shape of the section of the input to a layer that fully determines a pixel of the layer's output
    @staticmethod
    def _increase_patch_size(layer, input_patch=(1,1,1,1)):
        # Assumes that stride or subsample=1,
        # otherwise I should be doing something like last_layer.subsample[0]
        
        if hasattr(layer, 'subsample') and layer.subsample[0] != 1:
            print("Patch size doesn't currently account for layers with stride. Patches will be too small")
        
        if hasattr(layer, 'pool_size'):
            m = layer.pool_size # For pooling layers
            return [x[0]*x[1] for x in zip(input_patch, m)]
        elif hasattr(layer, 'filter_length'):
            # For 1D convolutional layers
            return (input_patch[0]+2*layer.filter_length,)
        elif hasattr(layer, 'nb_row') and hasattr(layer, 'nb_col'):
            return (input_patch[0]+2*layer.nb_row,
                    input_patch[1]+2*layer.nb_col)
        elif (hasattr(layer, 'kernel_dim1')
          and hasattr(layer, 'kernel_dim2')
          and hasattr(layer, 'kernel_dim3')):
            # 3D convolutional layer
            return (
                input_patch[0]+2*layer.kernel_dim1,
                input_patch[1]+2*layer.kernel_dim2,
                input_patch[2]+2*layer.kernel_dim3
            )
        else:
            return (1, 1,1,1) # Assume it's an activation layer or the like
        
    @staticmethod
    def calculate_patch_size(model, examined_layer):
        """
        Calculates an appropriate patch size for a max patch visualization.
        Isn't always reliable.
        """
        # Assumes the model is sequential
        # Works for up to 3 dimensions...
        from keras.models import Sequential
        if not isinstance(model, Sequential):
            print("Your model isn't sequential... cannot calculate patch size.\nUsing default patch size of (8,8)")
            return (8,8)

        i = len(model.layers)-1
        while model.layers[i] != examined_layer and i != 0:
            i-=1

        patch_size = (1,1,1)
        while i >= 0:
            m = MaxPatch._input_multiplier(model.layers[i])
            patch_size = [x[0]*x[1] for x in zip(patch_size, m)]
            i-=1
        return tuple(patch_size)






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


