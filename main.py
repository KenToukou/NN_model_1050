import numpy as np

from model_2 import TwoLayerNet

net = TwoLayerNet(input_size=784, hidden_size=100, out_put_size=10)
print(net._params["W1"].shape)
print(net._params["b1"].shape)
print(net._params["W2"].shape)
print(net._params["b2"].shape)
print("###################")
x = np.random.randn(100, 784)
t = np.random.rand(100, 10)
grads = net.numerical_decent(x, t)
print(net._params["W1"].shape)
print(net._params["b1"].shape)
print(net._params["W2"].shape)
print(net._params["b2"].shape)
