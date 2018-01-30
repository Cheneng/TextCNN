# -*- coding: utf-8 -*-

import pandas as pd
import string
import os

# the file dir
train_dir = '/Users/cc/Model/SentenceClassification/data/review/train.csv'
test_dir = '/Users/cc/Model/SentenceClassification/data/review/test.csv'

# read the data
train_data = pd.read_csv(train_dir, sep=None)

# check the data
#print(train_data.head())

# choose the column
clean_col = train_data['comment_text']

print(clean_col)



"""
for i, comment in enumerate(clean_col):
    for poun in string.punctuation:
        comment = comment.replace(poun, '')
        clean_col[i] = comment
        print(i)
print(train_data['comment_text'])
# 如何整体进行赋值？
"""


