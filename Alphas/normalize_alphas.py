from sklearn.preprocessing import StandardScaler, MinMaxScaler
import all_alphas
import yfinance as yf
import pandas as pd

def normalize_alphas(x):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit_transform(x)
    return x


def standardize_alphas(x):
    scaler = StandardScaler()
    scaler.fit_transform(x)
    return x


