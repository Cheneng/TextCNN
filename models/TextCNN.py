# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.word_num, config.word_embedding_dimension)
        self.conv3 = nn.Conv2d(1, 1, (3, config.word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, 1, (4, config.word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, 1, (5, config.word_embedding_dimension))
        self.linear1 = nn.Linear(3, config.label_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolution
        map1 = F.relu(self.conv3(x))
        map2 = F.relu(self.conv4(x))
        map3 = F.relu(self.conv5(x))

        # Pooling
        max1 = F.max_pool2d(map1, (self.config.sentence_max_size-3+1, 1))
        max2 = F.max_pool2d(map2, (self.config.sentence_max_size-4+1, 1))
        max3 = F.max_pool2d(map3, (self.config.sentence_max_size-5+1, 1))
        # capture and concatenate the features
        max_over_time = torch.cat((max1, max2, max3), -1)
        max_over_time = max_over_time.view(self.config.batch_size, 1, -1)

        # project the features to the labels
        out = self.linear1(max_over_time)
        out = out.view(-1, self.config.label_num)

        return out


if __name__ == '__main__':
    print('\nRunning the model.py...')
    config = Config()
    model = TextCNN(config)
