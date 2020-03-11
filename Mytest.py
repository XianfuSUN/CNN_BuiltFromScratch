import mlp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

dim = 28
inp = 28 * 56
outp = 10
hiddenlayers = 100
lr = 0.001
alpha = 0.1
epochs = 40


class Data:
    def __init__(self):
        self.trainX = np.load("data/trainX.npy")
        self.trainX = self.trainX / 255
        self.trainy = mlp.labels2onehot(np.load("data/trainy.npy"))
        self.trainy = np.where(self.trainy > 0, 1, 0)
        self.testX = np.load("data/testX.npy")
        self.testX = self.testX / 255
        self.testy = mlp.labels2onehot(np.load("data/testy.npy"))
        self.testy = np.where(self.testy > 0, 1, 0)
        self.cur_index1, self.cur_index2 = 0, 0

    def nextTrain(self, batchsize):
        next = self.cur_index1 + batchsize
        batch = [self.trainX[self.cur_index1: next],
                 self.trainy[self.cur_index1: next]]
        self.cur_index1 = next
        return batch

    def nextTest(self, batchsize):
        next = self.cur_index2 + batchsize
        batch = [self.testX[self.cur_index2: next],
                 self.testy[self.cur_index2: next]]
        self.cur_index2 = next
        return batch

    def getTrain(self, range):
        batch = [self.trainX[range[0]: range[1]], self.trainy[range[0]: range[1]]]
        return batch

    def getTest(self, range):
        batch = [self.testX[range[0]: range[1]], self.testy[range[0]: range[1]]]
        return batch


def testAccuracy(logits, y):
    num, correct = 0, 0
    for v1, v2 in zip(logits, y):
        num += 1
        if np.argmax(v1) == np.argmax(v2):
            correct += 1
    return correct/num


def train1layer(single, data):
    batchsize = 10
    num = int(60000 / batchsize)
    test_index = 0
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    for j in range(epochs):
        for i in range(num):  # i maximum to 6000
            batch = data.nextTrain(batchsize)
            # x in (1568, batchsize), y in (10, batchsize) one hot encoded
            x, label = batch[0].T, batch[1].T
            # feed the train data
            logits = single.forward(x)
            crossEntropy = mlp.SoftmaxCrossEntropyLoss()
            loss = crossEntropy.forward(logits, label)
            # update according to the train data
            loss_grad =crossEntropy.backward()
            single.backward(loss_grad)
            single.step()

        print("epoch:%d"%j)
        data.cur_index1 = 0

        # output other values
        batch = data.getTrain((test_index, test_index + 1000))
        trainx, trainy = batch[0].T, batch[1].T
        train_logits = single.forward(trainx)
        trainLoss = mlp.SoftmaxCrossEntropyLoss().forward(train_logits, trainy)
        print("train loss:", trainLoss)
        train_loss.append(trainLoss)
        acc_train = testAccuracy(train_logits.T, trainy.T)
        print("train accuracy: ", acc_train)
        data.cur_index1 = 0
        train_accuracy.append(acc_train)

        # feed forward the test data
        test_batch = data.getTest((test_index, test_index + 1000))
        testx, test_lable = test_batch[0].T, test_batch[1].T
        test_out = single.forward(testx)
        loss = mlp.SoftmaxCrossEntropyLoss().forward(test_out, test_lable)
        test_loss.append(loss)
        print("test loss: ", loss)
        acc_test = testAccuracy(test_out.T, test_lable.T)
        test_accuracy.append(acc_test)
        print("test accuracy:L ", acc_test)
        test_index = (test_index + 1000) % 10000

        single.zerograd()
    return train_loss, test_loss, train_accuracy, test_accuracy


def train2layer(double, data):
    print("train2layers")
    test_index = 0
    batchsize = 10
    num = int(60000 / batchsize)
    loss = 0
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    for j in range(epochs):
        for i in range(num):  # i maximum to 6000
            batch = data.nextTrain(batchsize)
            # x in (1568, batchsize), y in (10, batchsize) one hot encoded
            x, label = batch[0].T, batch[1].T
            # feed the train data
            logits = double.forward(x)
            crossEntropy = mlp.SoftmaxCrossEntropyLoss()
            loss = crossEntropy.forward(logits, label)
            # update according to the train data
            loss_grad = crossEntropy.backward()
            double.backward(loss_grad)
            double.step()

        print("epoch:%d" %j)
        train_loss.append(loss)
        print("train loss: ", loss)
        data.cur_index1 = 0

        # output other values
        batch = data.getTrain([test_index, test_index + 1000])
        trainx, trainy = batch[0].T, batch[1].T
        train_logits = double.forward(trainx)
        acc_train = testAccuracy(train_logits.T, trainy.T)
        print("train accuracy: ", acc_train)
        train_accuracy.append(acc_train)
        data.cur_index1 = 0

        # feed forward the test data
        test_batch = data.getTest([test_index, test_index + 1000])
        testx, test_lable = test_batch[0].T, test_batch[1].T
        test_out = double.forward(testx)
        loss = mlp.SoftmaxCrossEntropyLoss().forward(test_out, test_lable)
        test_loss.append(loss)
        print("test loss: ", loss)
        acc_test = testAccuracy(test_out.T, test_lable.T)
        test_accuracy.append(acc_test)
        print("test accuracy:L ", acc_test)
        test_index = (test_index + 1000) % 10000

        double.zerograd()
    return train_loss, test_loss, train_accuracy, test_accuracy


if __name__ == "__main__":
    data = Data()
    single = mlp.SingleLayerMLP(inp, outp, hiddenlayers, alpha, lr)
    double = mlp.TwoLayerMLP(inp, outp, [hiddenlayers, hiddenlayers], alpha, lr)
    if sys.argv[1] == "1":
        result = train1layer(single, data)
    else:
        result = train2layer(double, data)

    train_accuracy, test_accuracy = result[2], result[3]
    train_loss, test_loss = result[0], result[1]
    X = np.linspace(1, 40, 40, endpoint=True)
    plt.plot(X, train_loss, color="red", label="train loss")
    plt.plot(X, test_loss, color="blue", label="test loss")
    plt.legend(loc="upper left")
    plt.savefig("./loss_twoLayer.jpg", dpi=144)
    plt.show()

    plt.plot(X, train_accuracy, color="red", label="train accuracy")
    plt.plot(X, test_accuracy, color="blue", label="test accuracy")
    plt.legend(loc="upper left")
    plt.savefig("./accuracy_twoLayer,jpg", dpi=144)
    plt.show()





