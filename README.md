# Convolutional Neural Networks for Sentence Classification

> This repo implements the *Convolutional Neural Networks for Sentence Classification* (Yoon Kim) using PyTorch

![model_archi](./pictures/model_archi.png)

You should rewrite the Dataset class in the data/dataset.py and put your data in '/data/train' or any other directory. 

run by

```
python3 main.py --lr=0.01 --epoch=20 --batch_size=16 --gpu=0 --seed=0 --label_num=2			
```