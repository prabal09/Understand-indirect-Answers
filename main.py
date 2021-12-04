
circa = "circa-data.tsv"
# with open(circa) as circa:
#     rd = csv.reader(circa, delimiter="\t", quotechar='"')
#     cnt = 0
#     for row in rd:
#         print(row)
#         print(type(row))
#         if cnt == 10:
#             break
#         cnt+=1

import argparse
import os
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import numpy as np
import datetime
import pdb
from model import BERT,AutoTokenizer
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup,logging
from solver import train_epoch,eval_model
from dataloader import create_data_loader,get_dfs
cuda_device = 0
import torch.nn as nn
from collections import defaultdict
import warnings
manual_seed = 100
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)
os.environ['PYTHONHASHSEED'] = str(0)
import pandas as pd
import csv
if torch.cuda.is_available():
	torch.cuda.manual_seed(manual_seed)
	torch.cuda.manual_seed_all(manual_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main(config):
	logging.set_verbosity_error()
	tokenizer = config.tokenizer
	model = BERT()
	device = config.device
	model = model.to(device)
	BATCH_SIZE = config.batch_size
	EPOCHS = config.epochs
	MAX_LEN = config.max_len
	optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
	pdb.set_trace()
	df_train, df_dev, df_test = get_dfs(config)
	# print(df_train.head(3))

	train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
	val_data_loader = create_data_loader(df_dev, tokenizer, MAX_LEN, BATCH_SIZE)
	test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
	total_steps = len(train_data_loader) * EPOCHS

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=total_steps
	)

	loss_fn = nn.MSELoss().to(device)
	history = defaultdict(list)
	best_accuracy = 0

	for epoch in range(EPOCHS):
		print(f'Epoch {epoch + 1}/{EPOCHS}')
		print('-' * 10)

		train_acc, train_loss = train_epoch(
			model,
			train_data_loader,
			loss_fn,
			optimizer,
			device,
			scheduler,
			len(df_train)
		)

		print(f'Train loss {train_loss} accuracy {train_acc}')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--lr', type=float, default=1e-4)

	parser.add_argument('--model_id', type=int, default=0)
	parser.add_argument('--model_path', type=str, default='./model')

	parser.add_argument("--circa", type=str, default="circa-data.tsv", help="Path to data (CSV/TSV format)")
	# parser.add_argument("--circa_dev", type=str, default="C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv", help="Path to UNLI dev (CSV format)")
	# parser.add_argument("--circa_test", type=str, default="C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv", help="Path to UNLI test (CSV format)")
	parser.add_argument('--max_len', default = 80)

	parser.add_argument('--tokenizer', default=BertTokenizer.from_pretrained('bert-base-uncased'))
	parser.add_argument('--device',default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
	parser.add_argument('--setting', type=str, default="unmatched")
	config = parser.parse_args()
	main(config)