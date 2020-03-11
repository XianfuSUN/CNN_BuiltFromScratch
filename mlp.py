import numpy as np


def random_normal_weight_init(input, output):
    return np.random.normal(0, 1, (output, input))


def random_weight_init(input, output):
    b = np.sqrt(6) / np.sqrt(input + output)
    return np.random.uniform(-b, b, (output, input))


def zeros_bias_init(outd):
    return np.zeros((outd, 1))


def labels2onehot(labels):
    return np.array([[i == lab for i in range(10)] for lab in labels])


class Transform:
    # This is the base class. It does not need to be filled out.
    # A Transform class represents a (one input, one output) function done to some input x
    #   In symbolic terms,
    #   if f represents the transformation, out is the output, x is the input,
    #     out = f(x)
    # The forward operation is called via the forward function
    # The derivative of the operation is computed by calling the backward function
    #   In ML, we compute the loss as a function of the output, so
    #   there is some function q that takes in the output of this transformation
    #   and outputs the loss
    #     Loss = q(out) = q(f(x))
    #   Hence, by chain rule (using the notation discussed in recitation),
    #     grad_wrt_x = (dout_dx)^T @ grad_wrt_out (@ is matrix multiply)
    #   This is useful in backpropogation.
    def __init__(self):
        # this function in a child class used for initializing any parameters
        pass

    def forward(self, x):
        # x should be passed as column vectors
        pass

    def backward(self, grad_wrt_out):
        # this function used to compute the gradient wrt the parameters
        # and also to return the grad_wrt_x
        #   which will be the grad_wrt_out for the Transform that provided the x
        pass

    def step(self):
        # this function used to update the parameters
        pass

    def zerograd(self):
        # after you updated you might want to zero the gradient
        pass


class Identity(Transform):
    """
  Identity Transform
  This exists to give you an idea for how to fill out the template
  """

    def __init__(self):
        Transform.__init__(self)

    def forward(self, x, train=True):
        self.shape = x.shape
        return x

    def backward(self, grad_wrt_out):
        return np.ones(self.shape) * grad_wrt_out


class ReLU(Transform):
    # ReLU non-linearity
    # IMPORTANT the Autograder assumes these function signatures

    def __init__(self):
        Transform.__init__(self)
        self.shape = [0, 0]
        self.x = None

    def forward(self, x, train=True):
        self.shape = x.shape
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_wrt_out):
        dout_dx = np.where(self.x > 0, 1, 0)
        return dout_dx * grad_wrt_out


class LinearMap(Transform):
    # This represents the matrix multiplication step
    # IMPORTANT the Autograder assumes these function signatures
    def __init__(self, indim, outdim, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.indim = indim
        self.outdim = outdim
        self.w = random_weight_init(indim, outdim)
        self.b = zeros_bias_init(outdim)
        self.w_grad = None
        self.b_grad = None
        self.x = None
        self.lr = lr
        self.batchsize = 1
        self.alpha = alpha
        self.w_momentum, self.b_momentum = 0, 0

    def forward(self, x):
        # assumes x is a batch of column vectors (ie the shape is (indim,batch_size))
        # return shape (outdim,batch)
        self.batchsize = x.shape[1]
        self.x = x
        output = np.matmul(self.w, x) + self.b
        return output

    def zerograd(self):
        self.w_grad = np.zeros([self.outdim, self.indim])

    def backward(self, grad_wrt_out):
        # assumes grad_wrt_out is in shape (outdim,batch)
        # return shape indim,batch
        self.w_grad = np.matmul(grad_wrt_out, self.x.T)
        self.b_grad = np.sum(grad_wrt_out, axis=1).reshape(self.outdim, 1)
        output = np.matmul(grad_wrt_out.T, self.w).T
        return output

    def step(self):
        self.w_momentum = self.w_momentum * self.alpha + self.w_grad
        self.w = self.w - self.lr * self.w_momentum
        self.b_momentum = self.b_momentum * self.alpha + self.b_grad
        self.b = self.b - self.lr * self.b_momentum

    def getW(self):
        # return the weights
        return self.w

    def getb(self):
        # return the bias
        return self.b

    def loadparams(self, w, b):
        # IMPORTANT this function is called by the autograder
        # to set the weights and bias of this LinearMap
        self.w = w
        self.b = b


class SoftmaxCrossEntropyLoss:
    """
  Softmax Cross Entropy loss
  IMPORTANT the Autograder assumes these function signatures
  """
    def forward(self, logits, labels):
        # assumes the labels are one-hot encoded
        # assumes both logits and labels have shoape (num_classes,batch_size)
        # activated, shape(num_classes, batch_size)
        self.labels = labels
        self.batchsize = logits.shape[1]
        self.activated = np.exp(logits)/np.sum(np.exp(logits), axis=0)
        return -np.sum(labels * np.log(self.activated)) / self.batchsize

    def backward(self):
        # return shape (num_classes,batch_size)
        return (self.activated - self.labels) / self.batchsize


class SingleLayerMLP(Transform):
    # This MLP has one hidden layer
    # IMPORTANT the Autograder assumes these function signatures

    def __init__(self, inp, outp, hiddenlayer=100, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.inp = inp
        self.outp = outp
        self.hidden_units = hiddenlayer
        self.lr = lr
        self.alpha = alpha
        self.lm1 = LinearMap(inp, hiddenlayer, self.alpha)
        self.lm2 = LinearMap(hiddenlayer, outp, self.alpha)
        self.relu = ReLU()

    def forward(self, x, train=True):
        # x has shape (indim,batch)
        layer1 = self.lm1.forward(x)  # layer 1 hiddenp, batch
        activation = self.relu.forward(layer1)  # activation hidden, batch
        layer2 = self.lm2.forward(activation)  # layer 2 outp, batch
        return layer2

    def backward(self, grad_wrt_out):
        grad_1 = self.lm2.backward(grad_wrt_out)
        relu_grad = self.relu.backward(grad_1)
        grad_2 = self.lm1.backward(relu_grad)
        return grad_2

    def zerograd(self):
        self.lm1.zerograd()
        self.lm2.zerograd()

    def step(self):
        self.lm1.step()
        self.lm2.step()

    def loadparams(self, Ws, bs):
        # Ws is a length two list, representing the weights for the first LinearMap and the second LinearMap
        # ie Ws == [LinearMap1.W, LinearMap2.W]
        # bs is the bias for the layers, in the same order
        self.lm1.loadparams(Ws[0], bs[0])
        self.lm2.loadparams(Ws[1], bs[1])

    def getWs(self):
        # return the weights in a list, ordered [LinearMap1.W, LinearMap2.W]
        return [self.lm1.getW(), self.lm2.getW()]

    def getbs(self):
        # return the bias in a list, ordered [LinearMap1.b, LinearMap2.b]
        return [self.lm1.getb(), self.lm2.getb()]


class TwoLayerMLP(Transform):
    # This MLP has one hidden layer
    # IMPORTANT the Autograder assumes these function signatures

    def __init__(self, inp, outp, hiddenlayers=(100, 100), alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.inp = inp
        self.outp = outp
        self.hiddenlayers = hiddenlayers
        self.alpha = alpha
        self.lr = lr
        self.single = SingleLayerMLP(inp, hiddenlayers[1], hiddenlayers[0],
                                     self.alpha, self.lr)
        self.lm = LinearMap(hiddenlayers[1], outp, self.alpha, self.lr)
        self.relu = ReLU()

    def forward(self, x, train=True):
        layer1 = self.single.forward(x)
        relu = self.relu.forward(layer1)
        output = self.lm.forward(relu)
        return output

    def backward(self, grad_wrt_out):
        grad1 = self.lm.backward(grad_wrt_out)
        grad_relu = self.relu.backward(grad1)
        grad2 = self.single.backward(grad_relu)
        return grad2

    def zerograd(self):
        self.single.zerograd()
        self.lm.zerograd()

    def step(self):
        self.lm.step()
        self.single.step()

    def loadparams(self, Ws, bs):
        # Ws is a length three list, representing the weights for the first LinearMap
        # the second LinearMap, and the third
        # ie Ws == [LinearMap1.W, LinearMap2.W, LinearMap3.W]
        # bs is the bias for the layers, in the same order
        self.single.loadparams(Ws[:2], bs[:2])
        self.lm.loadparams(Ws[2], bs[2])

    def getWs(self):
        # return the weights in a list, ordered [LinearMap1.W, LinearMap2.W, LinearMap3.W]
        Ws = self.single.getWs()
        Ws.append(self.lm.getW())
        return Ws

    def getbs(self):
        # return the weights in a list, ordered [LinearMap1.b, LinearMap2.b, LinearMap3.b]
        bs = self.single.getbs()
        bs.append(self.lm.getb())
        return bs