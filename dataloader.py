import pdb
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
class Excerpt_Dataset(Dataset):

    def __init__(self, data, maxlen, tokenizer,type_ = 'q+a'):
        #Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen
        self.type_ = type_
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        #Select the sentence and label at the specified index in the data frame
        # pdb.set_trace()
        excerpt1 = self.df.loc[index, 'questionX']
        excerpt2 = self.df.loc[index, 'answerY']
        # print('premise',excerpt1)
        # print('hypothesis',excerpt2)
        target = self.df.loc[index, 'goldstandard2']
        identifier = self.df.loc[index, 'id']
        #Preprocess the text to be suitable for the transformer
        if self.type_ == 'a-only':
            tokens1 = self.tokenizer.tokenize("")
            tokens2 = self.tokenizer.tokenize(excerpt2)
        elif self.type_ == 'q-only':
            tokens1 = self.tokenizer.tokenize(excerpt1)
            tokens2 = self.tokenizer.tokenize("")
        else:
            tokens1 = self.tokenizer.tokenize(excerpt1)
            tokens2 = self.tokenizer.tokenize(excerpt2)

        tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']


        #Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        # print('target',target)
        target = torch.tensor([target], dtype=torch.float32)

        # print(target.size())
        # target = target.reshape((-1, 1))
        return input_ids, attention_mask, target

def get_dfs(config_path):
#    df = pd.read_csv(config.circa,encoding='utf8',engine='python')
    df = pd.read_csv(config_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
    df.dropna(subset=["goldstandard2"], inplace=True)
    # df.goldstandard1 = df.goldstandard1.map({'Yes': 1, 'No': 2,'Yes, subject to some conditions':3,'In the middle, neither yes nor no':4,'Other':5})
    df.goldstandard2 = df.goldstandard2.map({'Yes': 1, 'No': 2,'Yes, subject to some conditions':3,'In the middle, neither yes nor no':4,'Other':5})
    # df.goldstandard1 = pd.factorize(df.goldstandard1)[0] + 1
    # df.goldstandard2 = pd.factorize(df.goldstandard2)[0] + 1
#    df_dev = pd.read_csv(config.usnli_dev)
#    df_test = pd.read_csv(config.usnli_test)
    msk = np.random.rand(len(df)) < 0.6
    df_ = df[~msk]
    msk1 = np.random.rand(len(df_)) < 0.5
    df_train = df[msk]
    df_dev = df_[~msk1]
    df_test = df_[msk1]
    return df_train,df_dev,df_test
