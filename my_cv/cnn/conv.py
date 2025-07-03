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
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))

        for im_region, i, j, in self.iterate_regions(input):
            # element wise mult of the region and the filter, giving a 3d array
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        
        return output