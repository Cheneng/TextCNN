import pickle

class Helper(object):

    def __init__(self):

        self.word2ind = {}
        self.ind2word = {}

        word2ind_file = '/Users/cc/Model/SentenceClassification/data/review/processed_data/word2ind.pkl'
        ind2word_file = '/Users/cc/Model/SentenceClassification/data/review/processed_data/ind2word.pkl'

        with open(word2ind_file, 'rb') as f:
            self.word2ind = pickle.load(f)

        with open(ind2word_file, 'rb') as f:
            self.ind2word = pickle.load(f)

    def fun_word2ind(self, x):
        out = []

        for sentence in x:
            temp = []
            for word in sentence:
                if word in self.word2ind.keys():
                    temp.append(self.word2ind[word])
                else:
                    temp.append(self.word2ind['<oov>'])
            out.append(temp)

        return out


if __name__ == '__main__':
    helper = Helper()
    s = [['could', 'you', 'help', 'me', 'to', 'shit', 'this', 'movie']]
    print(helper.word2ind['thanks'])
    print(helper.fun_word2ind(s))

