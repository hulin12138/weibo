# _*_ coding=utf-8 -*_
'''
Created on 2017-03-21
@author: yyf
'''

import numpy as np

from tools import *

class Softmax:
    '''
    If sparse, then X contains examples in each row.
    else X contains examples in a column.
    '''

    def __init__(self, sparse, *args):
        '''
        The units in each layer.
        '''
        self.weights = []
        self.biases = []
        self.num_layers = len(args)
        self.sparse = sparse
        self.use_regular = False
        # Initialize randomly with small values.
        tot = np.sqrt(reduce(lambda a,b: a*b, args, 1))
        for idx in range(self.num_layers - 1):
            self.weights.append(np.random.random_sample(size=(args[idx + 1], args[idx])) / tot)
            # bias should be a column vector to broadcast column-wise.
            self.biases.append(np.random.random_sample(size=(args[idx + 1], 1)) / tot)

    def __softmax__(self, mat, axis=0):
        '''
        Calculate softmax function.
        '''
        mat = np.exp(mat - np.max(mat, axis=axis, keepdims=True))
        return mat / np.sum(mat, axis=axis, keepdims=True)

    def __sigmoid__(self, mat):
        return 1. / (1 + np.exp(-mat))

    def extend_label(self, Y):
        Y = np.array(Y)
        if len(Y.shape) == 1:
            ty = np.zeros((self.weights[-1].shape[0], len(Y)), dtype=np.float64)
            for c, r in enumerate(Y):
                ty[r, c] = 1
            Y = ty
        return Y

    def loss(self, O, Y):
        '''
        Cross entropy loss.
        :param O: Predicted output matrix with sample output in a column.
        '''
        Y = self.extend_label(Y)
        tl = np.mean(-Y.T.dot(np.log(O)))
        if not self.use_regular:
            return tl
        return reduce(lambda a, b: a + self.regular / 2. * np.sum(b **2), self.weights, tl)

    def forward_propagation(self, X):
        '''
        :param X: a small batch with example in each column
            sparse: if X is sparse with every column which is the indexes of non-zero,
        then we can speed up matrix multiply
        '''
        Z = [None]
        A = [X]
        for i in range(self.num_layers - 1):
            if self.sparse and i == 0:
                z = np.zeros(shape=(self.weights[0].shape[0], len(X)), dtype=np.float64)
                for r in range(len(X)):
                    for c in range(len(X[r])):
                        z[:, r] += self.weights[0][:, X[r][c]]
                Z.append(z + self.biases[i])
            else:
                Z.append(self.weights[i].dot(A[i]) + self.biases[i])
            if i == self.num_layers - 2:
                A.append(self.__softmax__(Z[-1]))
            else:
                A.append(self.__sigmoid__(Z[-1]))
        return Z, A

    def back_propagation(self, Z, A, Y):
        delta = range(self.num_layers)
        grad_w = range(self.num_layers - 1)
        grad_b = range(self.num_layers - 1)
        Y = self.extend_label(Y)
        delta[-1] = A[-1] - Y
        for i in reversed(range(self.num_layers - 1)):
            # skip the first layer of which the delta is not needed.
            if i != 0:
                delta[i] = self.weights[i].T.dot(delta[i+1]) * A[i] * (1 - A[i])
            if self.sparse and i == 0:
                grad_w[i] = np.zeros((delta[i+1].shape[0], self.weights[0].shape[1]), dtype=np.float64)
                for r in range(len(A[0])):
                    for c in range(len(A[0][r])):
                        grad_w[i][:, A[0][r][c]] += delta[i+1][:, r]
                grad_w[i] /= len(A[0])
            else:
                grad_w[i] = delta[i+1].dot(A[i].T) / A[i].shape[1]
            if self.use_regular:
                grad_w[i] += self.weights[i] * self.regular
            grad_b[i] = np.mean(delta[i+1], axis=1, keepdims=True)
        return delta, grad_w, grad_b

    def gradient_check(self, X, Y):
        epsilon = 1e-4
        Z, A = self.forward_propagation(X)
        delta, grad_w, grad_b = self.back_propagation(Z, A, Y)
        for idx, w in enumerate(self.biases):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    tmp = w[i, j]
                    w[i, j] = epsilon + tmp
                    Z, A = self.forward_propagation(X)
                    loss_plus = self.loss(A[-1], Y)
                    w[i, j] = tmp - epsilon
                    Z, A = self.forward_propagation(X)
                    loss_minus = self.loss(A[-1], Y)
                    w[i, j] = tmp
                    grad = (loss_plus - loss_minus) / (2 * epsilon)
                    if not abs(grad - grad_b[idx][i][j]) < epsilon:
                        print("gradient failed with i %d, j %d, idx %d" % (i, j, idx))
                        print(grad, grad_b[idx][i][j])
                        return
        for idx, w in enumerate(self.weights):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    tmp = w[i, j]
                    w[i, j] = epsilon + tmp
                    Z, A = self.forward_propagation(X)
                    loss_plus = self.loss(A[-1], Y)
                    w[i, j] = tmp - epsilon
                    Z, A = self.forward_propagation(X)
                    loss_minus = self.loss(A[-1], Y)
                    w[i, j] = tmp
                    grad = (loss_plus - loss_minus) / (2 * epsilon)
                    if not abs(grad - grad_w[idx][i][j]) < epsilon:
                        print("weights gradient failed with i %d, j %d, idx %d"%(i, j, idx))
                        print(grad, grad_w[idx][i][j])
                        return

    def fit(self, X, Y):
        '''
        :param X: Samples matrix with each example in a column
        :param Y: Samples label with each label in a column which is one-hot
        '''
        self.data_train = X
        self.data_label = np.array(Y, dtype=np.int32)

    def next_batch(self, batch_size):
        for bgn in range(0, len(self.data_train), batch_size):
            end = bgn + batch_size
            if end < len(self.data_train):
                yield self.data_train[bgn:end], self.data_label[bgn:end]

    def __train__step(self, batchx, batchy, learn_rate):
        Z, A = self.forward_propagation(batchx)
        delta, grad_w, grad_b = self.back_propagation(Z, A, batchy)
        for w, gw in zip(self.weights, grad_w):
            w -= learn_rate * gw
        for b, gb in zip(self.biases, grad_b):
            b -= learn_rate * gb

    def __early_stop__(self, precisions, epsilon):
        # print("in __early_stop__ std %f"%(np.std(precisions[-10:])))
        if len(precisions) > 3 and np.std(precisions[-3:]) < epsilon:
            return True
        return False

    def train(self, learn_rate=0.01, epsilon=1e-4, eval_epoch=50, batch_size=16, max_epoch=10000,\
              regular=None):
        assert (hasattr(self, "data_train"))
        if regular != None:
            print ("use regularization")
            self.regular = regular
            self.use_regular = True
        if not hasattr(self, "precision_log"):
            self.precision_log = []
        for i in range(max_epoch):
            if i % eval_epoch == 0 or i == max_epoch - 1:
                self.precision_log.append(self.precision())
                print("precision is %f after %d epoches" %(self.precision_log[-1], i))
                if self.__early_stop__(self.precision_log, epsilon):
                    print("finish traning")
                    return 0
            for batchx, batchy in self.next_batch(batch_size):
                self.__train__step(batchx, batchy, learn_rate)
        return -1

    def predict(self, X):
        Z, A = self.forward_propagation(X)
        return A[-1]

    def precision(self, X=None, Y=None, topk=1):
        if not X:
            X = self.data_train
            Y = self.data_label
        if not self.sparse:
            Y = np.argmax(Y, axis=0)
        indices = [Y, range(len(Y))]
        y_hat = self.predict(X)
        y_hat -= y_hat[indices]
        y_hat = np.sum(y_hat > 0, axis=0) < topk
        return np.sum(y_hat, dtype=np.float64) / len(Y)










