"""Functions used for data manipulations"""
from typing import Tuple
import numpy as np
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """Dataset class we will use for NN models."""
    def __init__(self, data: np.ndarray, past: int, noise: float = 1, train: bool = True) -> None:
        """Initialization.
        
        :param data:    data to be processed 
        :param past:    number of past time-steps used for prediction
        :param noise:   standard deviation of noise we wil apply to the data
        """
        self.data = data
        self.data_size = len(data)
        self.past = past 
        self.noise = noise
        self.train = train

    def __len__(self):
        'Denotes the total number of samples'
        return self.data_size - self.past

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Select sample
        if self.noise and self.train:
            noise = np.random.normal(0, self.noise, self.past)
            augmented_data = self.data[idx: idx + self.past] + noise
        else:
            augmented_data = self.data[idx: idx + self.past]
        return augmented_data, self.data[idx + self.past]


def get_dataloader(data: np.ndarray, test: float = 0.2, past: int = 2, batch: int = 64, noise: float = 1):
    """Generate train and test dataloaders.

    :param data:  data we use for generation
    :param test:  size of test data
    :param batch: batch size 
    :returns:     train and test dataloaders  
    """
    train_size = int(len(data) * (1 - test))
    train_data, test_data = data[:train_size], data[train_size:]
    
    training_data = TimeSeriesDataset(train_data, past, noise=noise)
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=False)

    test_data = TimeSeriesDataset(test_data, past, train=False, noise=noise)
    test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=False)
    
    return train_dataloader, test_dataloader