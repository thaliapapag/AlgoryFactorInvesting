
import numpy as np
import pandas as pd
import json
import os
import time
from tqdm import tqdm
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

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

# Alphas
def alpha10(data):
    delta_close = data['Close'].diff()
    ts_min = delta_close.rolling(window=4).min()
    ts_max = delta_close.rolling(window=4).max()
    condition1 = ts_min > 0
    condition2 = ts_max < 0
    alpha = np.where(condition1, delta_close, np.where(condition2, delta_close, -delta_close))
    return np.array(pd.Series(alpha).rank())


#### I am making the ones below here work. ####

# (sign(delta(volume, 1)) * (-1 * delta(close, 1)))

def alpha12(data): # Works - will have to drop the first day
    delta_volume = data['Volume'].diff()
    delta_close = data['Close'].diff()
    return np.sign(delta_volume) * (-1 * delta_close)

def alpha13(data): # drops the fist 4 values
    # (-1 * rank(covariance(rank(close), rank(volume), 5)))

    rank_close = data['Close'].rank()
    rank_volume = data['Volume'].rank()

    covariance = rank_close.rolling(window=5).cov(rank_volume)
    rank_covariance = covariance.rank()

    return -1 * rank_covariance




def alpha14(data):  # Will have to drop the first 9 days from the dataset each time because they are all nan due to the
                    # correlation being over the previous 10 days. This is expected, and we can offset this by including
                    # 9 extra days at the beginning of the dataset, specifically for this alpha

    # ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))

    delta_returns = ((data['Close'] - data['Open']) / data['Open']).diff(3)
    open_volume_correlation = data['Open'].rolling(window=10).corr(data['Volume'])
    return -1 * delta_returns.rank() * open_volume_correlation

def alpha15(data): # Works - will have to drop the first 6 days each time

    # (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))

    high_rank = data['High'].rank()
    volume_rank = data['Volume'].rank()
    correlation = high_rank.rolling(window=3).corr(volume_rank)
    rank_correlation = correlation.rolling(window=3).apply(lambda x: pd.Series(x).rank().iloc[-1])
    return -1 * rank_correlation.rolling(window=3).sum()










def main():
    
    end_date = datetime.now() - timedelta(days=50)
    start_date = end_date - timedelta(days=15)

    ticker = 'GOOG'

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    # print(data)

    print(alpha13(data))

if __name__ == "__main__":
    main()