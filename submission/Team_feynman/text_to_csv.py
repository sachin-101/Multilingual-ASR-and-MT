# converting dataset to csv format for tabular dataset of pytorch
import pandas as pd

path_en = "../data/fr-en/europarl-v7.fr-en.en" # put path to europarl-v7.fr-en.en
path_fr = "../data/fr-en/europarl-v7.fr-en.fr" # put path to europarl-v7.fr-en.fr
MAX_LEN = 80 # set the max length of the sentences in the dataset

europarl_en = open(path_en, encoding='utf-8').read().split('\n')
europarl_fr = open(path_fr, encoding='utf-8').read().split('\n')
dataset = pd.DataFrame({'fr': [line for line in europarl_fr], 'en': [line for line in europarl_en], })

dataset['fr_len'] = dataset.fr.str.count(' ')
dataset['en_len'] = dataset.en.str.count(' ')
dataset = dataset.query('fr_len < %d & en_len < %d'%(MAX_LEN, MAX_LEN))

train_set_split_ratio = 0.1
train_set, val_set = dataset[['fr', 'en']].iloc[:int(len(dataset)*(1-train_set_split_ratio)), :], dataset[['fr', 'en']].iloc[int(len(dataset)*(1-train_set_split_ratio)):, :]
train_set.to_csv('./train_set.csv', index=False)
val_set.to_csv('./val_set.csv', index=False)