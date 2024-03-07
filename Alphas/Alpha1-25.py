
import numpy as np
import pandas as pd
import json
import os
import time
from tqdm import tqdm

def alpha1(data):
    """
    Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    """
    returns = data['close'].pct_change()
    condition = returns < 0
    #If the condition is True (i.e. the closing price change rate is less than 0), the standard deviation 
    #of the 20-day sliding window is selected; If False, the closing price itself is selected
    data['signed_power'] = np.where(condition, data['close'].rolling(window=20).std(), data['close']) ** 2
    rank = data['signed_power'].rolling(window=5).apply(np.argmax) + 1  # add 1 because rank is 1-based
    alpha1 = rank - 0.5

    return alpha1

def alpha2(data):
    """
    Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    """
    delta_log_volume = np.log(data['volume']).diff(2)
    returns = (data['close'] - data['open']) / data['open']
    correlation = delta_log_volume.rank().rolling(window=6).corr(returns.rank())
    return -1 * correlation

def alpha3(data):
    """
    Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    """
    correlation = data['open'].rank().rolling(window=10).corr(data['volume'].rank())
    return -1 * correlation

def alpha4(data):
    """
    Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    """
    # 对每日的低价进行排名 rank base on the daily lowest price
    ranked_low = data['low'].rank()
    
    ts_rank = ranked_low.rolling(window=9).apply(lambda x: np.argsort(np.argsort(x))[-1] + 1)
    
    return -1 * ts_rank

def alpha5(data):
    """
    Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    """
    
    avg_vwap = data['vwap'].rolling(window=10).sum() / 10
    open_minus_avg_vwap_rank = (data['open'] - avg_vwap).rank()
    abs_close_minus_vwap_rank = abs(data['close'] - data['vwap']).rank()
    alpha5 = open_minus_avg_vwap_rank * (-1 * abs_close_minus_vwap_rank)
    return alpha5

def alpha6(data):
    """
    Alpha#6: (-1 * correlation(open, volume, 10))
    """
    correlation_open_volume = data['open'].rolling(window=10).corr(data['volume'])
    alpha6 = -1 * correlation_open_volume
    return alpha6


def alpha6(data):
    """
    Alpha#6: (-1 * correlation(open, volume, 10))
    """
    correlation_open_volume = data['open'].rolling(window=10).corr(data['volume'])

    alpha6 = -1 * correlation_open_volume
    return alpha6

def alpha7(data):
    adv20 = data['volume'].rolling(window=20).mean()
    delta_close_7 = data['close'].diff(7)
    condition = adv20 < data['volume']

    result = np.where(condition, -1 * data['close'].diff(7).abs().rolling(window=60).apply(lambda x: x.rank().iloc[-1]), -1)
    alpha7 = np.where(condition, result * np.sign(delta_close_7), result)
    return alpha7

def alpha8(data):
    sum_open_5 = data['open'].rolling(window=5).sum()
    sum_returns_5 = data['returns'].rolling(window=5).sum()
    combined_sum = sum_open_5 * sum_returns_5
    delayed_combined_sum = combined_sum.shift(10)
    final_expression = combined_sum - delayed_combined_sum
    alpha8 = -1 * final_expression.rank()
    return alpha8

def alpha9(data):
    delta_close = data['close'].diff()
    ts_min = delta_close.rolling(window=5).min()
    ts_max = delta_close.rolling(window=5).max()
    condition1 = ts_min > 0
    condition2 = ts_max < 0
    return np.where(condition1, delta_close, np.where(condition2, delta_close, -delta_close))

def alpha10(data):
    delta_close = data['close'].diff()
    ts_min = delta_close.rolling(window=4).min()
    ts_max = delta_close.rolling(window=4).max()
    condition1 = ts_min > 0
    condition2 = ts_max < 0
    alpha = np.where(condition1, delta_close, np.where(condition2, delta_close, -delta_close))
    return alpha.rank()

