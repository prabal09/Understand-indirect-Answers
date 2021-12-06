from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import pdb
import numpy as np
def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
            optimizer.zero_grad()

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            # target1 = torch.tensor(target, dtype=int)
            output,_ = model(input_ids=input_ids, attention_mask=attention_mask)
            # src = torch.zeros(output.shape).to(device)
            # src = src.scatter_(1, target1, 1)
            # pdb.set_trace()
            loss = criterion(output, target.reshape(output.shape))
            #.squeeze()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Training loss is {train_loss/len(train_loader)}")

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            count += 1

    return mean_loss/count

def get_rmse(output, target):
    err = torch.sqrt(metrics.mean_squared_error(target, output))
    return err

def predict(model, dataloader, device):
    predicted_label = np.array([])
    actual_label = np.array([])
    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output,index = model(input_ids, attention_mask)
            index = index.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            target.reshape(index.shape)
            # pdb.set_trace()
            predicted_label = np.append(predicted_label,index)
            actual_label = np.append(actual_label,target)

    return predicted_label,actual_label




def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    # pdb.set_trace()
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)