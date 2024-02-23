import pandas as pd
import numpy as np


def adv(hst, d):
    # average daily dollar volume for the past d days
    print(hst)
    vol = hst["Volume"]
    rolling_mean = vol.rolling(window=d).mean()
    hst["adv"] = rolling_mean
    return hst["adv"]


def vwap(hst):  # yf.Ticker.history object
    # daily volume-weighted average price
    vol = hst["Volume"]
    close = hst["Close"]

    sum_vol = vol.sum()

    hst["vwap"] = (vol * close).div(sum_vol)

    return hst["vwap"]


def ts_rank():
    pass


def decay_linear(hst, col, d):
    weights = np.array([i for i in range(d, 0, -1)])
    sum_weights = np.sum(weights)

    hst["weighted_ma"] = (
        hst[col]
        .rolling(window=d, center=True)
        .apply(lambda x: np.divide(np.sum(weights * x), sum_weights), raw=True)
    )

    return hst["weighted_ma"]


def correlation_days(x, y, d):
    # returns a scalar
    df = pd.DataFrame
    x_new = x.tail(d).values()
    y_new = y.tail(d).values()
    corr = np.corrcoef(x_new, y_new)

    return corr[0][1]  # return value from correlation matrix
