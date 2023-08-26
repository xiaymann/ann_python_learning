import numpy as np
import math

# softmax函数的实现
# 但是这样实现容易产生溢出问题
def softmax1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
b = softmax1(a)
print(b)

# 加上或减去输入信号的最大值可以避免溢出问题
def softmax2(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([1010, 1000, 990])
y = softmax2(a)
print(y)

# softmax函数的输出值总和为1
print(np.sum(b))
print(np.sum(y))