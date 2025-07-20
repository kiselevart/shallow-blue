import numpy as np

class MaxPool2:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output

  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input

# same thing but implemented with custom pool size
class MaxPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        h, w, _ = image.shape
        ps = self.pool_size
        stride = ps

        new_h = (h - ps) // stride + 1
        new_w = (w - ps) // stride + 1

        ps = self.pool_size

        for i in range(new_h):
            for j in range(new_w):
                start_h = i * stride
                end_h = start_h + ps
                start_w = j * stride
                end_w = start_w + ps

                im_region = image[start_h:end_h, start_w:end_w]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w, num_channels = input.shape
        ps = self.pool_size
        stride = ps

        new_h = (h - ps) // stride + 1
        new_w = (w - ps) // stride + 1
        output = np.zeros((new_h, new_w, num_channels))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):
      d_L_d_input = np.zeros(self.last_input.shape)

      for im_region, i, j in self.iterate_regions(self.last_input):
        h, w, f = im_region.shape
        amax = np.amax(im_region, axis=(0, 1))

        for i2 in range(h):
          for j2 in range(w):
            for f2 in range(f):
              if im_region[i2, j2, f2] == amax[f2]:
                d_L_d_input[i*2+i2, j*2+j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input