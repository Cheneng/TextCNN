# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import Config
from models import TextCNN
from data import Review
from utils import Helper

# 创建helper工具
helper = Helper()
config = Config(sentence_max_size=50, batch_size=2,
                word_num=len(helper.word2ind), label_num=7)
training_set = Review()
training_iter = data.DataLoader(dataset=training_set,
                                batch_size=config.batch_size, num_workers=2)

#print(training_set[0:2])
data, labes = training_set[0]
print(data)

# 创建模型
model = TextCNN(config)

# 设置多分类损失函数
criterion = nn.MultiLabelSoftMarginLoss()

# 设置优化器
optimizer = optim.SGD(model.parameters(), lr=config.lr)

loss = 0

for data, label in training_iter:
    print(data)


