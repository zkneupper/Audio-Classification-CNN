import numpy as np
from keras.layers import Layer
from keras.layers import InputSpec
from keras.utils import conv_utils
from keras.utils import np_utils

class RandomCropping1D(Layer):
    """Randomized cropping layer for 1D input (e.g. temporal sequence).
    It crops along the time dimension (axis 1).
    # Arguments
        cropping: int
            How many units long the cropped dimension (axis 1) will be.
            The value must be less than the original length of 
            axis 1.
    # Input shape
        3D tensor with shape `(batch, axis_to_crop, features)`
    # Output shape
        3D tensor with shape `(batch, cropped_axis, features)`
    """

    def __init__(self, cropping=None, **kwargs):
        super(RandomCropping1D, self).__init__(**kwargs)
        self.cropping = conv_utils.normalize_tuple(cropping, 1, 'cropping')
        self.input_spec = InputSpec(ndim=3)    

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            length = self.cropping[0]
        else:
            length = None
        return (input_shape[0],
                length,
                input_shape[2])

    def call(self, inputs):

        if self.cropping[0] >= inputs.shape[1]:
            raise ValueError('`cropping` length should be less than the original length '
                             '(' + str(inputs.shape[1]) + ')')
        
        n_shift_max = (inputs.shape[1] + 1) - self.cropping[0]
        left = np.random.randint(0, (n_shift_max))
        right = left + self.cropping[0]
        
        return inputs[:, left:right, :]        
        
    def get_config(self):
        config = {'cropping': self.cropping}
        base_config = super(RandomCropping1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
