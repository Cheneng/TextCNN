# —*- coding: utf-8 -*-


class Config(object):
    def __init__(self, word_embedding_dimension=100, word_num=20000,
                 epoch=2, sentence_max_size=40, cuda=False,
                 label_num=2, learning_rate=0.01, batch_size=1,
                 out_channel=100):
        self.word_embedding_dimension = word_embedding_dimension     # 词向量的维度
        self.word_num = word_num
        self.epoch = epoch                                           # 遍历样本次数
        self.sentence_max_size = sentence_max_size                   # 句子长度
        self.label_num = label_num                                   # 分类标签个数
        self.lr = learning_rate
        self.batch_size = batch_size
        self.out_channel=out_channel
        self.cuda = cuda
