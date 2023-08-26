# coding: utf-8
import sys, os
from os import path
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
# sys.path.append(os.path.dirname(path.dirname(__file__)))
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
print(train_size)
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(f"batch_mask is {batch_mask}")

x_batch = x_train[batch_mask]
print(f"x_batch is {x_batch}")
t_batch = t_train[batch_mask]
print(f"t_batch is {t_batch}")