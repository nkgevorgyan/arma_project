from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from utils import evalute_model
from data import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_ARMA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.float()
        return self.net(x)


class LSTM_ARMA(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size        
        self.relu = nn.ReLU()
        
        self.encoder = nn.LSTM(
            1, hidden_size, batch_first=True
        ).to(device)

        self.first_linear = nn.Linear(hidden_size, 4)
        self.second_layer = nn.Linear(4, 1)
        
    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        x = x.float()
        x = self.encoder(x)[1][0]
        return self.second_layer(self.relu(self.first_linear(x)))


def train_run(net, loader, optimizer, criterion):
    net.train()

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float()
        predictions = net(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_run(net, loader):
    net.eval()
    predictions = []
    target = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            predictions.append(net(X).reshape(len(y)))
            target.append(y)

        return evalute_model(torch.cat(target), torch.cat(predictions)), torch.cat(predictions)


def train_model(net, learning_rate, series, num_epoch, eval_step, batch, past, noise):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_dataloader, val_dataloader = get_dataloader(series, batch=batch, past=past, noise=noise)

    for epoch in range(1, num_epoch+1):
        net.to(device)
        train_run(net, train_dataloader, optimizer, criterion)
        if epoch % eval_step == 0:
            val_results, predictions = validate_run(net, val_dataloader)
            print(f"Epoch: {epoch} -- MAE: {val_results['mae']} -- RMSE: {val_results['rmse']}")
    
    return net, predictions