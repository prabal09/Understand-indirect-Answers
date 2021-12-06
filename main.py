import pdb

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import Excerpt_Dataset, get_dfs
from model import *
from solver import *
from checkpoint_load import *

MAX_LEN_TRAIN = 20
MAX_LEN_VALID = 20
MAX_LEN_TEST = 20
BATCH_SIZE = 8
LR = 1e-5
NUM_EPOCHS = 3
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'

if __name__ == "__main__":
    config_path = "C:/Users/praba/PycharmProjects/Understanding-Indirect-Answers/Understanding-Indirect-Answers/circa-data.tsv"
    df_train, df_dev, df_test = get_dfs(config_path)
    ## Configuration loaded from AutoConfig
    bert_config = AutoConfig.from_pretrained(MODEL_NAME)
    ## Tokenizer loaded from AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ## Creating the model from the desired transformer model
    # pdb.set_trace()
    model = Bert.from_pretrained(MODEL_NAME, config=bert_config,n_classes = len(set(df_train.goldstandard2)))
    ## GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Putting model to device
    model = model.to(device)
    ## Takes as the input the logits of the positive class and computes the binary cross-entropy
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss().to(device)
    # criterion = nn.BCELoss().to(device)
    ## Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=LR)
    type_ ='q+a'  #'a-only'; 'q+a'

    train_set = Excerpt_Dataset(data=df_train, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer,type_=type_)
    valid_set = Excerpt_Dataset(data=df_dev, maxlen=MAX_LEN_VALID, tokenizer=tokenizer,type_=type_)
    test_set = Excerpt_Dataset(data=df_test, maxlen=MAX_LEN_TEST, tokenizer=tokenizer,type_=type_)

    ## Data Loaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    checkpoint = 'model_' + str(NUM_EPOCHS) + '.pt'
    # print(len(train_loader))
    # if len(find_files(checkpoint)) !=0:
    #     model,optimizer,epoch,loss = load_checkpt(PATH = checkpoint,MODEL_NAME = 'bert-base-uncased')
    print("Training Model")
    train(model=model,
      criterion=criterion,
      optimizer=optimizer,
      train_loader=train_loader,
      val_loader=valid_loader,
      epochs = NUM_EPOCHS,
     device = device)
    # save_checkpt(model = model,optimizer = optimizer,EPOCH = NUM_EPOCHS,PATH = checkpoint)
    predicted_label,actual_label = predict(model,test_loader,device)
    pdb.set_trace()
    print("predicted label",predicted_label)
    print("actual_label",actual_label)
    target_names = ['Yes','No','Yes, subject to some conditions','In the middle, neither yes nor no','Other']
    print(classification_report(actual_label,predicted_label,target_names=target_names))
    # loss = evaluate(model, criterion, train_loader, device)
    # output = predict(model, test_loader, device)
    # out2 = []
    # for out in output:
    #     out2.append(out.cpu().detach().numpy())
    # out = np.array(out2).reshape(len(out2))
    # submission = pd.DataFrame({'id': df_test['id'], 'pre': df_test['pre'],'hyp':df_test['hyp'],'target':out})
    # submission.to_csv('submission.csv', index=False)

