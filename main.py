# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from config import Config
from models import TextCNN
from data import Review
import pickle

torch.manual_seed(1)

if torch.cuda.is_available():
    torch.cuda.set_device(2)

# 创建配置文件
if torch.cuda.is_available():
    config = Config(sentence_max_size=50,
                    batch_size=3000,
                    word_num=11000,
                    label_num=7,
                    learning_rate=0.1,
                    cuda=True,
                    epoch=1000)

else:
    config = Config(sentence_max_size=50,
                    batch_size=3,
                    word_num=11000,
                    label_num=7,
                    learning_rate=0.1)

# 创建Dataset和DataLoader
training_set = Review()
testing_set = Review(train=False)

if config.cuda:
    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=config.batch_size,
                                    num_workers=2,
                                    pin_memory=True)
    testing_iter = data.DataLoader(dataset=training_set,
                                   batch_size=config.batch_size,
                                   num_workers=2,
                                   pin_memory=True)

else:
    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=config.batch_size,
                                    num_workers=2)
    testing_iter = data.DataLoader(dataset=testing_set,
                                   batch_size=config.batch_size,
                                   num_workers=2)

# 创建模型
model = TextCNN(config)
embeds = nn.Embedding(config.word_num, config.word_embedding_dimension)

if config.cuda and torch.cuda.is_available():
    model.cuda()
    embeds = embeds.cuda()

# 设置多分类损失函数
criterion = nn.MultiLabelSoftMarginLoss()

# 设置优化器
optimizer = optim.SGD(model.parameters(), lr=config.lr)

# loss_sum用来记录前n个损失相加的结果，count作为计数变量
loss_sum = 0
count = 0

right = 0
sample_num = 0
count_vali = 0
loss_table = []
loss_pkl = './data/loss.pkl'

# Train the model
for epoch in range(config.epoch):

    for data, label in training_iter:

        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            labels = label.byte().cuda()

        optimizer.zero_grad()
        input_data = embeds(autograd.Variable(data))
        input_data = input_data.unsqueeze(1)

        out = model(input_data)

        if config.cuda:
            loss = criterion(out, autograd.Variable(label.float()).cuda())
#            pred = (F.sigmoid(out).data - 0.5 > 0)
#            compare = (label.byte().cuda() == pred.cuda())
        else:
            loss = criterion(out, autograd.Variable(label.float()))
#            pred = (F.sigmoid(out).data - 0.5 > 0)
#            compare = (label.byte() == pred)

        loss_sum += loss
        count += 1

        if count >= 10:
            count_vali += 1
            loss_table.append((loss_sum/(count*config.batch_size)).data[0])

            print("epoch", epoch, end='  ')
            print("The loss is: %.9f" % (loss_sum/(count*config.batch_size)).data[0])
            #print("The accuracy is: ", right, '/', sample_num, right/sample_num)

            loss_sum = 0
            count = 0
            #sample_num = 0
            #right = 0

        loss.backward()
        optimizer.step()

        torch.save(model.state_dict(), f='checkpoints/out.model'+str(epoch))

        with open(loss_pkl, 'wb') as f:
            pickle.dump(loss_table, f)

        if count_vali >= 10:
            count_vali = 0

            # Validation
            for voli_data, voli_labels in testing_iter:

                if config.cuda and torch.cuda.is_available():
                    voli_data = voli_data.cuda()
                    voli_labels = voli_labels.byte().cuda()

                voli_data = embeds(autograd.Variable(voli_data))
                voli_data = voli_data.unsqueeze(1)

                out = model(voli_data)
                pred = (F.sigmoid(out).data - 0.5 > 0)

                if config.cuda:
                    #loss = criterion(out, autograd.Variable(label.float()).cuda())
                    #pred = (F.sigmoid(out).data - 0.5 > 0)
                    compare = (label.byte().cuda() == pred.cuda())
                else:
                    #loss = criterion(out, autograd.Variable(label.float()))
                    #pred = (F.sigmoid(out).data - 0.5 > 0)
                    compare = (label.byte() == pred)

                for i in compare:
                    sample_num += 1
                    if i.all():
                        right += 1

            print(right, '/', sample_num)
            right = 0
            sample_num = 0

