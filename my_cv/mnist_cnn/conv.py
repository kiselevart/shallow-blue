import numpy as np

# Conv layer with 3x3 filters
class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with the dimensions (num_filters, 3, 3)
        # divide by 9 to reduce the variance of the initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding. - image is a 2d np array
        '''
        h, w = image.shape

        for i in range(h-2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input. Returns a 3d numpy array with dimensions (h, w, num_filters). - input is a 2d np array'''
        self.last_input = input 

        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))

        for im_region, i, j, in self.iterate_regions(input):
            # element wise mult of the region and the filter, giving a 3d array
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        
        return output

    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None