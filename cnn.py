import numpy as np
import os
import pickle as pk
import pdb
import time


# You are free to use any number of global helper functions
# We'll be testing your implementation of im2col and im2col_bw
def random_normal_weight_init(shape, channels):
    return np.random.normal(0, 0.1, (shape[0], channels, shape[1], shape[2]))


def random_weight_init(input, output, seed):
    np.random.seed(seed)
    b = np.sqrt(6) / np.sqrt(input + output)
    return np.random.uniform(-b, b, (input, output), )


# input shape c w h, filter shape, d w, h
def filter_weight_init(inputShape, filterShape, seed=0):
    np.random.seed(seed)
    output = (filterShape[0] + inputShape[0]) * filterShape[1] * filterShape[2]
    b = np.sqrt(6) / np.sqrt(output)
    return np.random.uniform(-b, b, (filterShape[0], inputShape[0],
                                     filterShape[1], filterShape[2]))


def zeros_bias_init(outd):
    return np.zeros((outd, 1))


def labels2onehot(labels):
    return np.array([[i == lab for i in range(10)] for lab in labels])


def im2col(X, k_height, k_width, padding=1, stride=1):
    # Construct the im2col matrix of intput feature map X.
    # Input:
    #   X: 4D tensor of shape [N, C, H, W], intput feature map
    #   k_height, k_width: height and width of convolution kernel
    # Output:
    #   cols: 2D array
    im = []
    shape = X.shape
    batchsize = shape[0]
    channels = shape[1]
    size = k_width * k_height * channels
    k1, k2 = shape[2] + 2 * padding, shape[3] + 2 * padding
    N = int(((k1 - k_width + 1) / stride) * ((k2 - k_height + 1) / stride))
    for i in range(batchsize):
        cur_im = []
        Y = np.pad(X[i],((0,0), (padding, padding), (padding, padding))
        , "constant", constant_values=0)  # padding one image the x
        offsetx, offsety = 0, 0
        while offsety + k_height <= k1:
            offsetx = 0
            while offsetx + k_width <= k2:
                slice = Y[..., offsety: offsety + k_height, offsetx: offsetx + k_width]
                cur_im.append(slice.reshape([size]))  # slice [channel, k_width, k_height]
                offsetx += stride
            offsety += stride
        cur_im = np.array(cur_im)  # curImage(K^2C, N)
        im.append(cur_im)
    result = np.array(im).transpose((1,0,2)).reshape(batchsize * N,size)
    return result.T  # im should be in shape batchsize, N, k_width*k_height*C


def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    # Map gradient w.r.t. im2col output back to the feature map.
    # Input:
    #   grad_X_col: 2D array shape[k*k*c, h*w*batch]
    #   X_shape: (N, C, H, W)
    #   k_height, k_width: height and width of convolution kernel
    # Output:
    #   X_grad: 4D tensor of shape X_shape, gradient w.r.t. feature map
    batch, channels = X_shape[0], X_shape[1]
    height, width = X_shape[2], X_shape[3]
    N = height * width
    k1, k2 = X_shape[2] + 2 * padding, X_shape[3] + 2 * padding
    x_grad_pad = np.zeros([batch, channels, k1, k2])
    grad_X_col = grad_X_col.reshape([channels, k_height * k_width, N, batch])
    grad_X_col = grad_X_col.transpose([2, 3, 0, 1]).reshape([N, batch, channels,
                                                             k_height, k_width])
    # add the gradients to the original matrix
    offsetx, offsety = 0, 0
    i = 0
    while offsety + k_height <= k1:
        offsetx = 0
        while offsetx + k_width <= k2:
            h, w = offsety + k_height, offsetx + k_width
            x_slide = x_grad_pad[..., offsety:h, offsetx:w]
            x_slide += grad_X_col[i]
            # x_grad_pad = addPartial(x_grad_pad, grad_X_col[i], offsetx, offsety)
            offsetx += stride
            i += 1
        offsety += stride
    # crop the padding
    x_grad = x_grad_pad[...,
             padding: padding + height,
             padding: padding + width]
    return x_grad


class ReLU:
    """
  ReLU non-linearity
  IMPORTANT the Autograder assumes these function signatures
  """

    def __init__(self):
        self.x = None

    def forward(self, x, train=True):
        # IMPORTANT the autograder assumes that you call np.random.uniform(0,1,x.shape) exactly once in this function
        if train:
            self.x = x
        return np.maximum(x, 0)

    def backward(self, dLoss_dout):
        dout_dx = np.where(self.x > 0, 1, 0)
        return dout_dx * dLoss_dout


class Conv:
    #
    # Class to implement convolutional layer
    # Arguments -
    #   1. input_shape => (channels, height, width)
    #   2. filter_shape => (num of filters, filter height, filter width)
    #   3. random seed
    #
    #   Initialize your weights and biases of correct dimensions here
    #   NOTE :- Initialize biases as 1 (NOT 0) of correct size. Inititalize bias as a 2D vector
    #   Initialize momentum

    def __init__(self, input_shape, filter_shape, rand_seed=0):
        self.input_shape = input_shape
        self.channels = input_shape[0]
        self.filterNum = filter_shape[0]
        self.filter_shape = filter_shape
        # filter in shape [numOfFilters, channels, w, h]
        self.filter = filter_weight_init(input_shape, filter_shape, rand_seed)
        self.bias = np.ones([self.filterNum, 1])
        # transformed inputs and filters
        self.x_transform = None
        self.filter_trans = None
        self.pad = 0
        self.stride = 1
        self.batchsize = 1
        # init gradients and momentums
        self.grad_w_mom, self.grad_b_mom = 0, 0
        self.grad_w, self.grad_b = 0, 0

    def forward(self, inputs, stride=1, pad=0):
        # Implement forward pass of convolutional operation here
        # Arguments -
        #   1. inputs => input image of dimension (batch_size, channels, height, width)
        #   2. stride => stride of convolution
        #   3. pad => padding
        #
        # Perform forward pass of convolution between input image and filters.
        # Return the output of convolution operation
        self.pad = pad
        self.stride = stride
        in_shape = inputs.shape
        self.batchsize = in_shape[0]
        height, width = in_shape[2], in_shape[3]
        # x_transform should be in shape[K_Height * kwidth*c, N * batchsize]
        self.x_transform = im2col(inputs, self.filter_shape[1],
                             self.filter_shape[2], pad, stride)
        # outcome should be in[numofkernels, N * batchsize]
        self.filter_trans = self.filter.reshape(self.filterNum,
                            self.channels * self.filter_shape[1] * self.filter_shape[2])
        outcome = np.matmul(self.filter_trans, self.x_transform) + self.bias.reshape([self.filterNum, 1])
        conv_out = outcome.reshape([self.filterNum, height, width, self.batchsize])
        conv_out = conv_out.transpose((3, 0, 1, 2))
        return conv_out

    def backward(self, dloss):
        # Implement backward pass of convolutional operation here
        # Arguments -
        #   1. dloss => derivative of loss wrt output loss shape(batch, channels, h, w)
        #
        # Perform backward pass of convolutional operation
        # Return [gradient wrt weights, gradient wrt bias, gradient wrt input] in this order
        # loss shape [batch, channel, w, h]
        shape = dloss.shape
        # transform the dloss into 2D form [D, h*w*B]
        loss_2d = dloss.transpose((1, 2, 3, 0)).reshape(shape[1],
                                                        shape[0] * shape[2] * shape[3])
        x_grad_col = np.matmul(self.filter_trans.T, loss_2d)  # ( k^2C, N*B)
        w_grad_col = np.matmul(loss_2d, self.x_transform.T)  # (D, K^2C)
        grad_w = w_grad_col.reshape([shape[1], self.channels,  # (D, C, K, K)
                                     self.filter_shape[1], self.filter_shape[2]])
        in_shape = [self.batchsize, self.channels, self.input_shape[1], self.input_shape[2]]
        grad_x = im2col_bw(x_grad_col, in_shape, self.filter_shape[1], self.filter_shape[2],
                           self.pad, self.stride)
        grad_b = np.sum(loss_2d, axis=1).reshape(shape[1], 1)  # (D, 1)
        self.grad_w, self.grad_b = grad_w/self.batchsize, grad_b/self.batchsize
        return grad_w, grad_b, grad_x

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        # Update weights and biases of convolutional layer in this function
        # Arguments -
        #   1. learning_rate
        #   2. momentum_coeff
        #
        # Make sure to scale your gradient according to the batch size
        # Update your weights and biases using respective momentums
        # No need to return any value from this function.
        # Update weights and bias within the class as class attributes
        self.grad_w_mom = momentum_coeff * self.grad_w_mom - \
                          learning_rate * (self.grad_w)
        self.grad_b_mom = momentum_coeff * self.grad_b_mom - \
                          learning_rate * (self.grad_b)
        self.filter += self.grad_w_mom
        self.bias += self.grad_b_mom

    def get_wb_conv(self):
        return self.filter, self.bias


class MaxPool:
    # Class to implement MaxPool operation
    # Arguments -
    #   1. filter_shape => (filter_height, filter_width)
    #   2. stride
    def __init__(self, filter_shape, stride):
        self.filter_shape = filter_shape
        self.stride = stride
        self.mapping = None
        self.input_shape = None

    def forward(self, inputs):
        # Implement forward pass of MaxPool layer
        # Arguments -
        #   1. inputs => inputs to maxpool forward are outputs from conv layer
        #
        # Implement the forward pass and return the output of maxpooling operation on inputs
        f_h, f_w = self.filter_shape[0], self.filter_shape[1]
        in_h, in_w = inputs.shape[2], inputs.shape[3]
        self.input_shape = inputs.shape
        pool_h = int((in_h - f_h) / self.stride + 1)
        pool_w = int((in_w - f_w) / self.stride + 1)
        batch, channels = inputs.shape[0], inputs.shape[1]
        out = np.zeros([batch*channels, pool_h, pool_w])
        inputs = inputs.reshape([batch * channels, in_h, in_w])
        self.mapping = np.zeros(inputs.shape)
        self.mapping = self.mapping.reshape((batch * channels, in_h, in_w))
        for h in range(pool_h):
            for w in range(pool_w):
                h1, w1 = h * self.stride, w * self.stride
                h2, w2 = h1 + f_h, w1 + f_w
                slide = inputs[:, h1: h2, w1: w2]  # [b*c, k, k]
                slide_trans = slide.reshape((batch * channels, f_h * f_w))
                out[:, h, w] = np.max(slide_trans, axis=1)
                arg_max = np.argmax(slide_trans, axis=1)
                for i in range(batch * channels):
                    index = arg_max[i]
                    index1, index2 = h1 + index // f_w, w1 + index % f_w
                    self.mapping[i, index1, index2] = 1
        self.mapping = self.mapping.reshape((batch, channels, in_w, in_h))
        return out.reshape((batch, channels, pool_h, pool_w))

    def backward(self, dloss):
        # Implement the backward pass of MaxPool layer
        # Arguments -
        #   1. dloss => derivative loss wrt output
        #
        # Return gradient of loss wrt input of maxpool layer
        x_grad = np.zeros(self.input_shape)
        offsetx, offsety = 0, 0
        in_h, in_w = self.input_shape[2], self.input_shape[3]
        batch, channel = self.input_shape[0], self.input_shape[1]
        f_h, f_w = self.filter_shape[0], self.filter_shape[1]
        while offsety + f_h <= in_h:
            offsetx = 0
            while offsetx + f_w <= in_w:
                slide = x_grad[..., offsety: offsety + f_h,
                               offsetx: offsetx + f_w]
                index1, index2 = offsetx//self.stride, offsety//self.stride
                loss = dloss[..., index2, index1]
                slide += dloss[..., index2, index1].reshape([batch, channel, 1, 1])
                offsetx += self.stride
            offsety += self.stride
        grad_inputs = self.mapping * x_grad
        return grad_inputs


class LinearLayer:
    # Class to implement Linear layer
    # Arguments -
    #   1. input_neurons => number of inputs
    #   2. output_neurons => number of outputs
    #   3. rand_seed => random seed
    #
    # Initialize weights and biases of fully connected layer
    # NOTE :- Initialize bias as 1 (NOT 0) of correct dimension. Inititalize bias as a 2D vector
    # Initialize momentum for weights and biases
    def __init__(self, input_neurons, output_neurons, rand_seed=0):
        self.inps_shape = input_neurons
        self.oups_shape = output_neurons
        self.W = random_weight_init(input_neurons, output_neurons, rand_seed)
        self.b = np.ones([output_neurons, 1])
        self.x = None
        self.batchsize = 0
        self.grad_w = 0
        self.grad_b = 0
        self.w_mom = 0
        self.b_mom = 0

    def forward(self, features):
        # Implement forward pass of linear layer
        # Arguments -
        #   1. features => inputs to linear layer[batch, indim]
        #
        # Perform forward pass of linear layer using features and weights and biases
        # Return the result
        # NOTE :- Make sure to check the dimension of bias
        self.batchsize = features.shape[0]
        self.x = features
        result = np.matmul(features, self.W) + self.b.reshape(self.oups_shape)
        return result

    def backward(self, dloss):
        # Implement backward pass of linear layer
        # Arguments -
        #   dloss => gradient of loss wrt outputs
        #   dloss [batch, outdim]
        # Return [gradient of loss wrt weights, gradient of loss wrt bias, gradient of loss wrt input] in that order
        grad_x = np.matmul(dloss, self.W.T)
        grad_w = np.matmul(dloss.T, self.x).T
        grad_b = np.sum(dloss, axis=0).reshape(self.oups_shape, 1)
        self.grad_w = grad_w / self.batchsize
        self.grad_b = grad_b / self.batchsize
        return grad_w, grad_b, grad_x

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        # Implement this function to update the weights and biases of linear layer
        # Arguments -
        #   1. learning_rate
        #   2. momentum_coeff
        #
        # Update the weights and biases. No need to return any values
        self.w_mom = self.w_mom * momentum_coeff - learning_rate * self.grad_w
        self.W += self.w_mom
        self.b_mom = self.b_mom * momentum_coeff - learning_rate * self.grad_b
        self.b += self.b_mom

    def get_wb_fc(self):
        return self.W, self.b


class SoftMaxCrossEntropyLoss:
    # Class to implement softmax and cross entropy loss
    def __init__(self):
        self.labels = 0
        self.batchsize = 0
        self.activated = 0

    def forward(self, logits, labels, get_predictions=False):
        # Forward pass through softmax and loss function
        # Arguments -
        #   1. logits => pre-softmax scores
        #   2. labels => true labels of given inputs
        #   3. get_predictions => If true, the forward function returns predictions along with the loss
        #
        # Return negative cross entropy loss between model predictions and true values
        self.labels = labels
        self.batchsize = logits.shape[0]
        self.activated = np.exp(logits) /\
                         np.sum(np.exp(logits), axis=1).reshape((self.batchsize, 1))
        predict = np.argmax(self.activated, axis=1)
        loss = -np.sum(labels * np.log(self.activated))
        if get_predictions:
            return loss, predict
        return loss

    def backward(self):
        # gradient of loss with respect to inputs of softmax
        return self.activated - self.labels


class ConvNet:
    # Class to implement forward and backward pass of the following network -
    # Conv -> Relu -> MaxPool -> Linear -> Softmax
    # For the above network run forward, backward and update
    def __init__(self):
        # Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        # Conv of input shape 3x32x32 with filter size of 1x5x5
        # then apply Relu
        # then perform MaxPooling with a 2x2 filter of stride 2
        # then initialize linear layer with output 20 neurons
        # Initialize SotMaxCrossEntropy object
        self.conv = Conv((3, 32, 32), (1, 5, 5))
        self.relu = ReLU()
        self.maxpool = MaxPool((2, 2), 2)
        self.lm = LinearLayer(16*16, 20)
        self.loss = SoftMaxCrossEntropyLoss()
        self.batch = 0

    def forward(self, inputs, y_labels):
        # Implement forward function and return loss and predicted labels
        # Arguments -
        # 1. inputs => input images of shape batch x channels x height x width
        # 2. labels => True labels
        # return loss, pred_labels
        self.batch = inputs.shape[0]
        conv1 = self.conv.forward(inputs, stride=1, pad=2)
        relu1 = self.relu.forward(conv1)
        maxpool1 = self.maxpool.forward(relu1)
        lm1 = self.lm.forward(maxpool1.reshape(self.batch, 16*16))
        loss, predict = self.loss.forward(lm1, y_labels, get_predictions=True)
        print(loss, predict)
        return loss, predict

    def backward(self):
        # Implement this function to compute the backward pass
        # Hint: Make sure you access the right values returned from the forward function
        loss_grad = self.loss.backward()  # batch, oudim
        lm_grad = self.lm.backward(loss_grad)  # batch, indim
        lm_grad_trans = lm_grad[2].reshape((self.batch, 1, 16, 16))
        maxpool_grad = self.maxpool.backward(lm_grad_trans)
        relu_grad = self.relu.backward(maxpool_grad)
        conv_gard = self.conv.backward(relu_grad)

    def update(self, learning_rate, momentum_coeff):
        # Implement this function to update weights and biases with the computed gradients
        # Arguments -
        # 1. learning_rate
        # 2. momentum_coefficient
        self.lm.update(learning_rate, momentum_coeff)
        self.conv.update(learning_rate, momentum_coeff)


if __name__ == '__main__':
    # You can implement your training and testing loop here.
    # We will not test your training and testing loops, however, you can generate results here.
    # NOTE - Please generate your results using the classes and functions you implemented.
    # DO NOT implement the network in either Tensorflow or Pytorch to get the results.
    # Results from these libraries will vary a bit compared to the expected results
    pass

