# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class TextCNN(BasicModule):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.out_channel = config.out_channel
        self.conv3 = nn.Conv2d(1, 1, (3, config.word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, 1, (4, config.word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, 1, (5, config.word_embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size-5+1, 1))
        self.linear1 = nn.Linear(3, config.label_num)

    def forward(self, x):
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.config.label_num)

        return x


if __name__ == '__main__':
    print('running the TextCNN...')