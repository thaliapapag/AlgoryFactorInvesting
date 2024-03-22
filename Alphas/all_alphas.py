import yfinance as yf
from datetime import timedelta
from datetime import datetime
import numpy as np
from sklearn import linear_model
import pandas as pd
import json
import os
import time
#import pandas_market_calendars as mcal
#from tqdm import tqdm

#methods - Thalia
'''
def adjust_date(date):

    calendar = mcal.get_calendar('NYSE')
    
    # Check if the given date is a trading day
    # If it's not a trading day, find the next trading day
    while not calendar.valid_days(start_date=date, end_date=date):
        date += pd.Timedelta(days=1)
    
    return date 
'''
def adjust_date(date):
    while date.weekday() >= 5:
        date = date + timedelta(days=1)
    
    if date.month == 1 and date.day == 1:
        date = date + timedelta(days =1)
    
    while date.weekday() >= 5:
        date = date + timedelta(days=1)

    # Monday is 0 and Sunday is 6
    return date  

#value of x d days ago
def delay(x, d, date = datetime.now().date()):

    if(not date == datetime.now().date()):

        new_date = x.index[x == date].max()

        print("NEW START DATE", new_date)

        date = adjust_date(new_date)
    else:
        date = adjust_date(x.index[-1])
    most_recent_date = date  # Get the most recent date from the index
    x_days_ago = most_recent_date - timedelta(days=d)

    adj_date = adjust_date(x_days_ago)
    col = x.loc[adj_date]  # Use .loc[] to retrieve the row corresponding to the date
    return col



#time-series max over the past d days 
def ts_max(x, d):
    max = x[:-d].max()
    return max

 
#which day ts_max(x, d) occurred on
def ts_argmax(x, d):
    x = x.tail(d)
    max_index = x.idxmax()
    max_value = x[max_index]
    return max_value


#which day ts_min(x, d) occurred on
def ts_argmin(x, d):
    x = x.tail(d)
    min_index = x.idxmin()
    min_value = x[min_index]
    return min_value


#time-series min over the past d days
def ts_min(x, d):
    d = int(d)
    min = x[-d:].min()
    print(min)
    return min

#time-series rank in the past d days
def ts_rank(x, d):
    data = np.array(x[:-d])
    df = pd.DataFrame(data)
    ranked_df = df.rank(axis=1, ascending=False)
    return ranked_df

#cross-sectional rank 
def rank(x):
    if isinstance(x, (float, int)):
    # Handle single value
        return x

    # Convert to pandas Series/DataFrame if necessary
    if not isinstance(x, (pd.Series, pd.DataFrame)):
        x = pd.Series(x)
    
    df = pd.DataFrame(x)
    ranked_df = df.rank(axis=1, ascending=False)
    return ranked_df

#time-serial correlation of x and y for the past d days 
def correlation(x, y, d):
    correlation = x.rolling(window=d).corr(y).dropna()

    return correlation

#today’s value of x minus the value of x d days ago 
def delta(x, d):
    delt = x[-1] - x[-d]
    return delt

#daily close-to-close returns 
def r(x):
    x_return = x["Close"].diff()
    x_return = x_return.dropna()
    return x_return

#adv{d} = average daily dollar volume for the past d days
def adv(x, d):
    x['DollarVolume'] = x['Close'] * x['Volume']
    
    # Sum the dollar volume over the past d days
    past_d_days_dollar_volume = x['DollarVolume'].tail(d).sum()
    

    # Calculate average daily dollar volume
    average_daily_dollar_volume = past_d_days_dollar_volume / d
    
    return average_daily_dollar_volume


#weighted moving average over the past d days with linearly decaying weights d, d – 1, ..., 1 (rescaled to sum up to 1)
def decay_linear(x, d):
    weights = [i + 1 for i in range(d, 0, -1)]
    weights = np.array(weights) / np.sum(weights)
    wma = np.convolve(x, weights[::-1], mode='valid')
    return wma

def scale(x, a=1):
    sum_abs_x = sum(abs(val) for val in x)
    scaled_x = [val * (a / sum_abs_x) for val in x]
    return scaled_x

#Alpha#51: 
#TODO: THIS COMES BACK AS AN ARRAY- AVERAGE AND SUM?
#(delay(close, 20) - delay(close, 10)) / 10) - calculating average daily price change over those ten days
#((delay(close, 10) - close) / 10) - calculating price change over more recent ten days 
#subtracts more recent from earlier 
#WAY TO MEASURE MOMENTUM OF SECURITY
#POS VALUE MEANS SLOWING MOMENTUM NEG VALUE MEANS INCREASING MOMENTUM
#((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
def alpha51(close):
    if (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05):
        alpha51= 1 
    else : 
        alpha51 = ((-1 * 1) * (close - delay(close, 1)))
    return alpha51

#Alpha#52
#identifies trading signals based on combo of price momentum, relative strength over medium to long term and recent trading activity 
#HIGH VALUES BUY, LOW VALUES SELL
# ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
def alpha52(low, returns, volume):
    alpha52 = ((((-1 * ts_min(low, 5)) + delay(low, 5, ts_min(low, 5))) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    return alpha52

#Alpha#53:
#BUY if switches from neg to pos 
#SELL if swtiched from pos to neg
# (-1 * delta((((close - low) - (high - close)) / (close - low)), 9)) 
def alpha53(close, low, high):
    alpha53 = (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    return alpha53

#Alpha#54: (idk if I like this one)
# BUY could be indicated by pos values 
# SELL could be indicated by neg values 
# ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
def alpha54(low, close, open, high):
    alpha54 = ((-1 * ((low - close) * (open**5))) / ((low - high) * (close**5))) 
    return alpha54


#Alpha#55: 
# BUY could be indicated by pos values 
# SELL could be indicated by neg values 
#(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
def alpha55(close, low, high, volume):
    alpha55 = (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    return alpha55 


#WENDYS ALPHAS
#Alpha#1: 
#(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
def alpha1(data):
    returns = data['Close'].pct_change()
    condition = returns < 0
    #If the condition is True (i.e. the closing price change rate is less than 0), the standard deviation 
    #of the 20-day sliding window is selected; If False, the closing price itself is selected
    data['signed_power'] = np.where(condition, data['Close'].rolling(window=20).std(), data['Close']) ** 2
    rank = data['signed_power'].rolling(window=5).apply(np.argmax) + 1  # add 1 because rank is 1-based
    alpha1 = rank - 0.5

    return alpha1

#Alpha#2: 
#(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
def alpha2(data):
    delta_log_volume = np.log(data['Volume']).diff(2)
    returns = (data['Close'] - data['Open']) / data['Open']
    correlation = delta_log_volume.rank().rolling(window=6).corr(returns.rank())
    return -1 * correlation

# Alpha#3: 
#(-1 * correlation(rank(open), rank(volume), 10))
def alpha3(data):
    correlation = data['Open'].rank().rolling(window=10).corr(data['Volume'].rank())
    return -1 * correlation

#Alpha#4: 
#(-1 * Ts_Rank(rank(low), 9))
def alpha4(data):
    # 对每日的低价进行排名 rank base on the daily lowest price
    ranked_low = data['Low'].rank()
    
    ts_rank = ranked_low.rolling(window=9).apply(lambda x: np.argsort(np.argsort(x))[-1] + 1)
    
    return -1 * ts_rank


#Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
def alpha5(data):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    # Calculate the volume-weighted average price
    vwap = (typical_price * data['Volume']).sum() / data['Volume']
    avg_vwap = vwap.rolling(window=10).sum() / 10
    open_minus_avg_vwap_rank = (data['Open'] - avg_vwap).rank()
    abs_close_minus_vwap_rank = abs(data['Close'] - vwap).rank()
    alpha5 = open_minus_avg_vwap_rank * (-1 * abs_close_minus_vwap_rank)
    return alpha5

#Alpha#6: 
#(-1 * correlation(open, volume, 10))
def alpha6(data):
    correlation_open_volume = data['Open'].rolling(window=10).corr(data['Volume'])
    alpha6 = -1 * correlation_open_volume
    return alpha6

#Alpha #7
#((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
def alpha7(data):
    adv20 = data['Volume'].rolling(window=20).mean()
    delta_close_7 = data['Close'].diff(7)
    condition = adv20 < data['Volume']

    result = np.where(condition, -1 * data['Close'].diff(7).abs().rolling(window=60).apply(lambda x: x.rank().iloc[-1]), -1)
    alpha7 = np.where(condition, result * np.sign(delta_close_7), result)
    return alpha7

#Alpha #8
#(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))) 
def alpha8(data):
    sum_open_5 = data['Open'].rolling(window=5).sum()
    sum_returns_5 = data['Returns'].rolling(window=5).sum()
    combined_sum = sum_open_5 * sum_returns_5
    delayed_combined_sum = combined_sum.shift(10)
    final_expression = combined_sum - delayed_combined_sum
    alpha8 = -1 * final_expression.rank()
    return alpha8

#Alpha 9:
#((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))) 
def alpha9(data):
    delta_close = data['Close'].diff()
    ts_min = delta_close.rolling(window=5).min()
    ts_max = delta_close.rolling(window=5).max()
    condition1 = ts_min > 0
    condition2 = ts_max < 0
    return np.where(condition1, delta_close, np.where(condition2, delta_close, -delta_close))

#Alpha 10:
#rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))) 
def alpha10(data):
    delta_close = data['Close'].diff()
    ts_min = delta_close.rolling(window=4).min()
    ts_max = delta_close.rolling(window=4).max()
    condition1 = ts_min > 0
    condition2 = ts_max < 0
    alpha = np.where(condition1, delta_close, np.where(condition2, delta_close, -delta_close))
    return alpha.rank()


def main():
    ticker = yf.Ticker("^GSPC")
    data = ticker.history(period="2mo") #pricing data for the S&P 500

    low = data['Low']
    volume = data['Volume'] #THIS REFERS TO VOLUME NOT PRICE VOLUME- should it be changed????
    returns = r(data)   
    close = data['Close']
    high = data['High']
    open = data['Open']
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    # Calculate the volume-weighted average price
    vwap = (typical_price * data['Volume']).sum() / data['Volume']

    #print(alpha51(close))
    print(alpha52(low, returns, volume))
    print('************************')
    print(alpha53(close, low, high))
    print(alpha54(low, close, open, high))
    print(alpha55(close, low, high, volume))

    print(alpha1(data))
    print(alpha2(data))
    print(alpha3(data))
    print(alpha4(data))
    print(alpha5(data))
    print(alpha6(data))
    print(alpha7(data))
    #print(alpha8(data))
    #print(alpha9(data)) #shouldn't return a numpy array should return a dataframe
    #print(alpha10(data)) #shouldn't return a numpy array should return a dataframe 



if __name__ == "__main__":
    main()
