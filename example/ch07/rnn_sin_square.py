# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:37:21 2020

@author: chaos
"""

import sys

import numpy as np
# import matrixslow as ms
from scipy import signal


# 构造正弦波和方波两类样本的函数 
def get_sequence_data(dimension=10, length=10,
                      number_of_examples=1000, train_set_ratio=0.7, seed=42):
    """
    生成两类序列数据
    正弦波和方波
    seq[0] sin
    seq[1] square
    """
    seq = [np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1),
          np.array(signal.square(np.arange(0, 10, 10 / length))).reshape(-1, 1)]

    data = []
    for i in range(2):
        s = seq[i]
        for j in range(number_of_examples // 2):
            sequence = s + np.random.normal(0, 0.6, (len(s), dimension))  # 加入噪声
            labels = np.array([int(i == k) for k in range(2)])
            data.append(np.c_[sequence.reshape(1, -1), labels.reshape(1, -1)])

    # 把各个类别的样本合在一起
    data = np.concatenate(data, axis=0)

    # 随机打乱样本顺序
    np.random.shuffle(data)

    # 计算训练样本数量
    train_set_size = int(number_of_examples * train_set_ratio)  # 训练集样本数量

    # 将训练集和测试集、特征和标签分开
    train_set = data[:train_set_size, :-2].reshape(-1, length, dimension)
    train_label = data[:train_set_size, -2:]
    test_set = data[train_set_size:, :-2].reshape(-1, length, dimension)
    test_label = data[train_set_size:, -2:]
    return train_set, train_label, test_set, test_label


# 构造RNN
seq_len = 96  # 序列长度
dimension = 16  # 输入维度
status_dimension = 12  # 状态维度

signal_train, label_train, signal_test, label_test = get_sequence_data(length=seq_len, dimension=dimension)

sys.path.append('../..')
from example.ch07.rnn_simple import SimpleRNN
import torch.nn.functional as F
import torch

if __name__ == '__main__':
    net = SimpleRNN(dimension, status_dimension)
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(50):

        # train
        for i, s in enumerate(signal_train):
            optimizer.zero_grad()

            inputs = np.array([np.mat(s[j], dtype=float) for j in range(seq_len)])
            x = torch.from_numpy(inputs)
            label = torch.from_numpy(np.mat(label_train[i, :]))
            out = F.softmax(net(x), dim=1)

            loss = F.mse_loss(out, label[0])
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.item()))

        # predict
        res = []
        for i, s in enumerate(signal_test):
            inputs = np.array([np.mat(s[j], dtype=float) for j in range(seq_len)])
            x = torch.from_numpy(inputs)
            out = F.softmax(net(x), dim=1)
            out = torch.ravel(out)
            res.append(out.detach().numpy())

        predict = np.array(res).argmax(axis=1)
        real = label_test.argmax(axis=1)
        accuracy = (real == predict).astype(np.int64).sum() / len(signal_test)
        print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, accuracy))


# 输入向量节点
# inputs = [ms.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(seq_len)]

# 输入权值矩阵
# U = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
# W = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
# b = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# last_step = None
# for iv in inputs:
#     h = ms.ops.Add(ms.ops.MatMul(U, iv), b)
#
#     if last_step is not None:
#         h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)
#
#     h = ms.ops.ReLU(h)
#
#     last_step = h
#
# fc1 = ms.layer.fc(last_step, status_dimension, 40, "ReLU")
# fc2 = ms.layer.fc(fc1, 40, 10, "ReLU")
# output = ms.layer.fc(fc2, 10, 2, "None")
#
# predict = ms.ops.SoftMax(output)
#
# label = ms.core.Variable((2, 1), trainable=False)
#
# loss = ms.ops.CrossEntropyWithSoftMax(output, label)
#
# learning_rate = 0.005
# optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)
#
# batch_size = 16
#
# for epoch in range(50):
#
#     batch_count = 0
#     for i, s in enumerate(signal_train):
#
#         for j, x in enumerate(inputs):
#             x.set_value(np.mat(s[j]).T)
#
#         label.set_value(np.mat(label_train[i, :]).T)
#
#         optimizer.one_step()
#
#         batch_count += 1
#         if batch_count >= batch_size:
#             print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))
#
#             optimizer.update()
#             batch_count = 0
#
#     pred = []
#     for i, s in enumerate(signal_test):
#
#         for j, x in enumerate(inputs):
#             x.set_value(np.mat(s[j]).T)
#
#         predict.forward()
#         pred.append(predict.value.A.ravel())
#
#     pred = np.array(pred).argmax(axis=1)
#     true = label_test.argmax(axis=1)
#
#     accuracy = (true == pred).astype(np.int).sum() / len(signal_test)
#     print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, accuracy))
