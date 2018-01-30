# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from config import Config
from models import TextCNN

config = Config()

# 创建模型
model = TextCNN(config)

# 设置损失函数
criterion = nn.CrossEntropyLoss()
"""
# 多分类损失函数，使用软交叉熵
criterion = nn.
"""

criterion = nn.MultiLabelSoftMarginLoss()

# 设置优化器
optimizer = optim.SGD(model.parameters(), lr=config.lr)


embed = nn.Embedding(20, config.word_embedding_dimension)

loss = 0

for i in range(40000):

    optimizer.zero_grad()

    data = autograd.Variable(torch.LongTensor([[1, 4, 6, 7, 2], [2, 3, 1, 2, 5], [3, 4, 5, 6, 7]]))
    data = embed(data)

    # 使用卷积的四个维度分别为 [batch_size, channel, sequence_length, embedding_size]
    data = data.view(config.batch_size, 1, -1, config.word_embedding_dimension)

    labels = autograd.Variable(torch.LongTensor([0, 1, 0]))

    out = model.forward(data)

#    print(out)

    loss = criterion(out, labels)

    if i % 1000 == 0:
        print('The loss is :', loss.data[0])

        _, pred = torch.max(out, 1)
        print('the training label is:' , pred.data[0], pred.data[1], pred.data[2])

    loss.backward()
    optimizer.step()

