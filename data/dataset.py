from torch.utils import data
import numpy as np
import pickle


class Review(data.Dataset):

    def __init__(self, train=True):

        training_file = '/Users/cc/Model/SentenceClassification/data/review/processed_data/train_split_list.pkl'
        labels_file = '/Users/cc/Model/SentenceClassification/data/review/processed_data/labels_array.pkl'

        self.train_set = []
        self.labels = np.array([])

        with open(training_file, 'rb') as f:
            self.train_set = pickle.load(f)

        with open(labels_file, 'rb') as f:
            self.labels = pickle.load(f)

        # 训练集则选取前80%，交叉验证选后20%
        if train is True:
            self.train_set = self.train_set[:int(0.8*len(self.train_set))]
            self.labels = self.labels[:int(0.8*len(self.labels))]

        else:
            self.train_set = self.train_set[int(0.8*len(self.train_set)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def __getitem__(self, index):
        return self.train_set[index], self.labels[index]

    def __len__(self):
        return len(self.train_set)


if __name__ == '__main__':
    x = Review()
    print(len(x))
    print(x.labels.shape)

