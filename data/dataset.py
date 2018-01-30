from torch.utils import data
import random
import os

"""
class ReviewPolor(data.Dataset):

    def __init__(self, pos_root, neg_root, train=True):
        
        #获取数据所在路径，并划分 "训练集" 和 "测试集"。
        
        self.review = [os.path.join(pos_root, file_name) for file_name in os.listdir(pos_root)]
        neg_passage = [os.path.join(neg_root, file_name) for file_name in os.listdir(neg_root)][:800]

        # 训练集则选取前80%，测试选后20%
        if train is True:
            self.review = self.review[:int(0.8*len(self.review))]
            neg_passage = neg_passage[:int(0.8*len(self.review))]
        else:
            self.review = self.review[int(0.8*len(self.review)):]
            neg_passage = neg_passage[int(0.8*len(self.review)):]

        # 合并且打乱序列
        self.review.extend(neg_passage)
        random.shuffle(self.review)

    def __getitem__(self, index):
        x = open(self.review[index], 'r')
        label = 1 if 'pos' in self.review[index].split('/') else 0
        out = x.readlines()
        return out, label

    def __len__(self):
        return len(self.review)

def prepare_data(passage, word2id):
    pass
"""


if __name__ == '__main__':
    data_dir = '/Users/cc/Model/SentenceClassification/data/review_polarity/txt_sentoken/'

