import os  # noqa
import sys  # noqa

import numpy as np

from .functions import cross_entropy_error, numerical_gradient, sigmoid, sof_max


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, out_put_size, wieght_init_std=0.01):
        self._params = {}
        self._params["W1"] = wieght_init_std * np.random.randn(input_size, hidden_size)
        self._params["b1"] = np.zeros(hidden_size)
        self._params["W2"] = wieght_init_std * np.random.randn(
            hidden_size, out_put_size
        )
        self._params["b2"] = np.zeros(out_put_size)

    def predict(self, x):
        W1, W2 = self._params["W1"], self._params["W2"]
        b1, b2 = self._params["b1"], self._params["b2"]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = sof_max(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_decent(self, x, t):
        loss_W = lambda W: self.loss(x, t)  # noqa
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self._params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self._params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self._params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self._params["b2"])
        return grads
