# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        #self.embedding = nn.Embedding(config.word_num, config.word_embedding_dimension)
        self.conv3 = nn.Conv2d(1, 1, (3, config.word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, 1, (4, config.word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, 1, (5, config.word_embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size-5+1, 1))
        self.linear1 = nn.Linear(3, config.label_num)

    def forward(self, x):
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
        x = x.view(self.config.batch_size, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.config.label_num)

        return x


if __name__ == '__main__':
    print('\nTesting the model.py...')

    torch.manual_seed(1)

    config = Config(batch_size=2, sentence_max_size=7)
    model = TextCNN(config)

    # Fake data
    embeds = nn.Embedding(200, 100)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


    # Fake data
    for epoch in range(10):

        x_data = [[2, 3, 1, 4, 3, 1, 2], [3, 12, 45, 6, 67, 88, 55]]
        y_data = [0, 0]
        x = autograd.Variable(torch.LongTensor(x_data))
        y = autograd.Variable(torch.LongTensor(y_data))

        x = embeds(x)
        x = x.unsqueeze(1)

        out = model(x)
        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()

        out = F.softmax(out, dim=1)
        out, ind = torch.max(out, dim=1)

        print("loss is: ", loss.data[0])
        print("The label is: [%d %d]" % (ind.data[0], ind.data[1]))
        optimizer.step()



