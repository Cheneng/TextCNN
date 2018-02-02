# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import Config
from models import TextCNN
from data import Review

torch.manual_seed(1)

# 创建配置文件
config = Config(sentence_max_size=50, batch_size=2,
                word_num=11000, label_num=7)

# 创建Dataset和DataLoader
training_set = Review()

if config.cuda:
    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=config.batch_size,
                                    num_workers=2,
                                    pin_memory=True)
else:
    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=config.batch_size,
                                    num_workers=2)

# 创建模型
model = TextCNN(config)
if config.cuda:
    model.cuda()

# 设置多分类损失函数
criterion = nn.MultiLabelSoftMarginLoss()

# 设置优化器
optimizer = optim.SGD(model.parameters(), lr=config.lr)

embeds = nn.Embedding(config.word_num, config.word_embedding_dimension)

loss_sum = 0
count = 0

for data, label in training_iter:

    optimizer.zero_grad()
    input_data = embeds(autograd.Variable(data))
    #print(input_data)

    input_data = input_data.unsqueeze(1)

    out = model(input_data)
    loss = criterion(out, autograd.Variable(label.float()))

    loss_sum += loss
    count += 1
    if count >= 1000:
        print("The loss is:", loss_sum/1000)
        loss_sum = 0
        count = 0

    #loss.backward()
    #optimizer.step()



