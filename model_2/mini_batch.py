# import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# ミニバッチ法
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)


train_loss_list: list = []
train_acc_list = []
test_acc_list = []


train_size = x_train.shape[0]
iters_num = 10000
batch_size = 100
learning_rate = 0.1


network = TwoLayerNet(input_size=784, hidden_size=50, out_put_size=10)
iter_per_epoch = max(train_size / batch_size, 1)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]
    grad = network.gradient(x_batch, t_batch)
    for key in ("W1", "b1", "W2", "b2"):
        network._params[key] -= learning_rate * grad[key]
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


markers = {"train": "o", "test": "s"}
x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label="train acc")
# plt.plot(x, test_acc_list, label="test acc", linestyle="--")
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc="lower right")
# plt.savefig("./out/learning.png")
# plt.show()
