#  Writing LeNet5 from Scratch in PyTorch - testing with MNIST.

 LeNet = one of the earliest Convolutional Neural Networks ever introduced.
 Proposed by Yann LeCun and others in 1998.
 Used for the recognition of greyscale handwritten characters.
 Input = greyscale image of 32x32.

 MNIST Training dataset -> 60,000 images
 MNIST Testing dataset -> 10,000 images



 How LeNet Works:
       7 layer Convolutional Neural Network
       Input -> should contain just one channel.
			 
    1. First convolutional layer.
        Filter size of 5x5 with 6 such filters.
        Will reduce the width and height of the image while increasing
        The depth (number of channels).
        Output would be 28x28x6.
				
    2. Pooling is applied.
        Decreases the feature map by half, i.e, 14x14x6
        Same filter size (5x5) with 16 filters is now applied to the
        output followed by a pooling layer.
        Reduces output feature map to 5x5x16
				
    3. Convolutional layer.
        Size 5x5 with 120 filters applied to flatten the feature map to 120
        values.
        Output is 10x10x16.
				
    4. Second pooling layer.
        Identical to previous one, but this time layer has 16 filters.
        Output is 5x5x16.
				
    5. Last convolutional layer.
        120 5x5 filters.
        Input is size 5x5x16 and filters are 5x5, output is 1x1x120.
        As a result, layers 4 and 5 are fully-connected.
        Also why in some implementations of LeNet-5 actually use a
        full-connected layer instead of a convolutional one as 5th layer.
        Reason for keeping this one as convolutional is the fact that if
        the input to the network is larger than the one used in 1 (initial
        input, so 32x32 in this case), this layer will not be a fully-
        connected one, as the output of each filter would not be 1x1.
				
    6. First fully-connected layer.
        Takes input of 120 neurons and returns 84 neurons.
				
    7. Last dense layer.
        Outputs 10 neurons, since MNIST data have 10 classes for each
        of the represented 10 numerical digits


 Victoria Tran
 Created using helper articles and videos - self taught.
 Using for teaching purposes.
