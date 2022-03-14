"""Helper functions"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_process import arma_generate_sample


def generate_series(
    p: int, 
    q: int, 
    time_steps: int, 
    sigma: float = 1
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generates time series based on ARMA process.
    
    :param p:            order of AR model
    :param q:            order of MA model
    :param time_steps:   number ot time-steps in the seies
    :param sigma:        standard deviation of the noise
    :returns:            values of time-series and ARMA coefficients 
    """
    # generate random AR and MA coefficients
    np.random.seed(0)
    ar_coeff = np.append([1], np.random.uniform(-1, 1, p))
    ma_coeff = np.append([1], np.random.uniform(-1, 1, q))
    series = arma_generate_sample(ar_coeff, ma_coeff, time_steps, sigma)
    return series, (ar_coeff, ma_coeff)


def evalute_model(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Calculate MAE and RMSE for the derived predictions
    
    :param prediction:   predicted values of time series
    :param target:       true value of time series
    :returns:            dictionary with calculated metrics
    """
    mae  = mean_absolute_error(target, prediction)
    rmse = np.sqrt(mean_squared_error(target, prediction))
    return ({'mae': mae, 'rmse': rmse})


def prepare_dataset(dataset: np.ndarray, past: int, test: float = 0.2):
    """Transform time series to the 2d array with shape (n_sample, n_previous_observations)
    
    :param dataset: time series we will use 
    :param past:    number of previouse observations will be used for the prediction
    :param test:    size of the test data
    :returns:       2d arrays with train and test data with corresponding labels
    """   
    X, y = [], []
    train_size = int(len(dataset) * (1 - test))
    
    for i in range(train_size - past):
        X.append(dataset[i: i + past])
        y.append(dataset[i + past])
    X_train = np.stack(X, axis=0)
    y_train = np.array(y)
    X, y = [], []

    for i in range(train_size, len(dataset) - past):
        X.append(dataset[i: i + past])
        y.append(dataset[i + past])
    X_test = np.stack(X, axis=0)
    y_test = np.array(y)
    
    return (X_train, y_train), (X_test, y_test)