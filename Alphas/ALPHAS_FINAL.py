import yfinance as yf
from datetime import timedelta
from datetime import datetime
import numpy as np
from sklearn import linear_model
import pandas as pd
import json
import os
import time
import normalize_alphas
import csv_to_dataframe
#import pandas_market_calendars as mcal
#from tqdm import tqdm

# methods - Thalia


def adjust_date(date):
    while date.weekday() >= 5:
        date = date + timedelta(days=1)

    if date.month == 1 and date.day == 1:
        date = date + timedelta(days=1)

    while date.weekday() >= 5:
        date = date + timedelta(days=1)

    # Monday is 0 and Sunday is 6
    return date


# value of x d days ago
def delay(x, d, date=datetime.now().date()):

    if not date == datetime.now().date():

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


# time-series max over the past d days
def ts_max(x, d):
    max = x[:-d].max()
    return max


# which day ts_max(x, d) occurred on
def ts_argmax(x, d):
    x = x.tail(d)
    max_index = x.idxmax()
    max_value = x[max_index]
    return max_value


# which day ts_min(x, d) occurred on
def ts_argmin(x, d):
    x = x.tail(d)
    min_index = x.idxmin()
    min_value = x[min_index]
    return min_value


# time-series min over the past d days
def ts_min(x, d):
    d = int(d)
    min = x[-d:].min()
    print(min)
    return min


# time-series rank in the past d days
def ts_rank(x, d):
    data = np.array(x[:-d])
    df = pd.DataFrame(data)
    ranked_df = df.rank(axis=1, ascending=False)
    return ranked_df


# cross-sectional rank
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


# time-serial correlation of x and y for the past d days
def correlation(x, y, d):
    correlation = x.rolling(window=d).corr(y).dropna()

    return correlation


# today’s value of x minus the value of x d days ago
def delta(x, d):
    delt = x[-1] - x[-d]
    return delt


# daily close-to-close returns
def r(x):
    x_return = x["Close"].diff()
    x_return = x_return.dropna()
    return x_return


# adv{d} = average daily dollar volume for the past d days
def adv(x, d):
    x["DollarVolume"] = x["Close"] * x["Volume"]

    # Sum the dollar volume over the past d days
    past_d_days_dollar_volume = x["DollarVolume"].tail(d).sum()

    # Calculate average daily dollar volume
    average_daily_dollar_volume = past_d_days_dollar_volume / d

    return average_daily_dollar_volume


# weighted moving average over the past d days with linearly decaying weights d, d – 1, ..., 1 (rescaled to sum up to 1)
def decay_linear(x, d):
    weights = [i + 1 for i in range(d, 0, -1)]
    weights = np.array(weights) / np.sum(weights)
    wma = np.convolve(x, weights[::-1], mode="valid")
    return wma


def scale(x, a=1):
    sum_abs_x = sum(abs(val) for val in x)
    scaled_x = [val * (a / sum_abs_x) for val in x]
    return scaled_x




# Alpha#54: (idk if I like this one)
# BUY could be indicated by pos values
# SELL could be indicated by neg values
# ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
def alpha54(data):
    low = data['Low']
    close = data['Close']
    open = data['Open']
    high = data['High']
    alpha54 = (-1 * ((low - close) * (open**5))) / ((low - high) * (close**5))
    return alpha54



# WENDYS ALPHAS
# Alpha#1:
# (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
def alpha1(data):
    returns = data["Close"].pct_change()
    condition = returns < 0
    # If the condition is True (i.e. the closing price change rate is less than 0), the standard deviation
    # of the 20-day sliding window is selected; If False, the closing price itself is selected
    data["signed_power"] = (
        np.where(condition, data["Close"].rolling(window=20).std(), data["Close"]) ** 2
    )
    rank = (
        data["signed_power"].rolling(window=5).apply(np.argmax) + 1
    )  # add 1 because rank is 1-based
    alpha1 = rank - 0.5

    return alpha1


# Alpha#2:
# (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
def alpha2(data):
    delta_log_volume = np.log(data["Volume"]).diff(2)
    returns = (data["Close"] - data["Open"]) / data["Open"]
    correlation = delta_log_volume.rank().rolling(window=6).corr(returns.rank())
    return -1 * correlation


# Alpha#3:
# (-1 * correlation(rank(open), rank(volume), 10))
def alpha3(data):
    correlation = data["Open"].rank().rolling(window=10).corr(data["Volume"].rank())
    return -1 * correlation


# Alpha#4:
# (-1 * Ts_Rank(rank(low), 9))
def alpha4(data):
    # 对每日的低价进行排名 rank base on the daily lowest price
    ranked_low = data["Low"].rank()

    ts_rank = ranked_low.rolling(window=9).apply(
        lambda x: np.argsort(np.argsort(x))[-1] + 1
    )

    return -1 * ts_rank


# Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
def alpha5(data):
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    # Calculate the volume-weighted average price
    vwap = (typical_price * data["Volume"]).sum() / data["Volume"]
    avg_vwap = vwap.rolling(window=10).sum() / 10
    open_minus_avg_vwap_rank = (data["Open"] - avg_vwap).rank()
    abs_close_minus_vwap_rank = abs(data["Close"] - vwap).rank()
    alpha5 = open_minus_avg_vwap_rank * (-1 * abs_close_minus_vwap_rank)
    return alpha5


# Alpha#6:
# (-1 * correlation(open, volume, 10))
def alpha6(data):
    correlation_open_volume = data["Open"].rolling(window=10).corr(data["Volume"])
    alpha6 = -1 * correlation_open_volume
    return alpha6

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


#These alphas contain external changing data over the hold period --> currently 2 months
def alphaExt1(data):
    start_data = data.index[0]
    end_data = data.index[-1]
    #technology select sector SPDR fund 
    tech_data = yf.Ticker('XLK').history(start=start_data, end=end_data)
    tech = csv_to_dataframe.create_y(tech_data)
    return tech

def alphaExt2(data):
    start_data = data.index[0]
    end_data = data.index[-1]
    #Financial select sector SPDR fund
    tech_data = yf.Ticker('XLF').history(start=start_data, end=end_data)
    tech = csv_to_dataframe.create_y(tech_data)
    return tech

def alphaExt3(data):
    start_data = data.index[0]
    end_data = data.index[-1]
    #Health Care Select Sector SPDR Fund 
    tech_data = yf.Ticker('XLV').history(start=start_data, end=end_data)
    tech = csv_to_dataframe.create_y(tech_data)
    return tech

def alphaExt4(data):
    start_data = data.index[0]
    end_data = data.index[-1]
    #Voliatility Index
    tech_data = yf.Ticker('VIXY').history(start=start_data, end=end_data)
    tech = csv_to_dataframe.create_y(tech_data)
    return tech

def alphaExt5(data):
    start_data = data.index[0]
    end_data = data.index[-1]
    #S&P 500 Index
    tech_data = yf.Ticker('SPY').history(start=start_data, end=end_data)
    tech = csv_to_dataframe.create_y(tech_data)
    return tech

def alphaExt6(data):
    start_data = data.index[0]
    end_data = data.index[-1]
    #Dow Jones Industrial Average
    tech_data = yf.Ticker('DIA').history(start=start_data, end=end_data)
    tech = csv_to_dataframe.create_y(tech_data)
    return tech

def alphaExt7(data):
    start_data = data.index[0]
    end_data = data.index[-1]
    #energy select sector SPDR fund
    tech_data = yf.Ticker('XLE').history(start=start_data, end=end_data)
    tech = csv_to_dataframe.create_y(tech_data)
    return tech

def vwap(data):
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    # Calculate the volume-weighted average price
    vwap = (typical_price * data["Volume"]).sum() / data["Volume"]
    return vwap

def alphas_to_df(data):
    #TODO: add more alphas to this list

    #before calculating alphas we need to standardize and normalize the data
    data = normalize_alphas.normalize_alphas(data)
    data = normalize_alphas.standardize_alphas(data)

    alpha1_df = alpha1(data)
    alpha2_df = alpha2(data)
    alpha3_df = alpha3(data)
    alpha4_df = alpha4(data)
    alpha5_df = alpha5(data)
    alpha6_df = alpha6(data)
    alpha12_df = alpha12(data)
    alpha13_df = alpha13(data)
    alpha14_df = alpha14(data)
    alpha15_df = alpha15(data)
    alphaExt1_df = alphaExt1(data)
    alphaExt2_df = alphaExt2(data)
    alphaExt3_df = alphaExt3(data)
    alphaExt4_df = alphaExt4(data)
    alphaExt5_df = alphaExt5(data)
    alphaExt6_df = alphaExt6(data)
    alphaExt7_df = alphaExt7(data)
    alpha54_df = alpha54(data)
    
    alphas_df = pd.concat([alpha1_df, alpha2_df, alpha3_df, alpha4_df, alpha5_df, alpha6_df, alpha12_df, alpha13_df, alpha14_df, alpha15_df, alpha54_df, alphaExt1_df, alphaExt2_df, alphaExt3_df, alphaExt4_df, alphaExt5_df, alphaExt6_df, alphaExt7_df], axis=1)
    alphas_df.columns = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha12', 'alpha13', 'alpha14', 'alpha15', 'alpha54', 'alphaExt1', 'alphaExt2', 'alphaExt3', 'alphaExt4', 'alphaExt5', 'alphaExt6', 'alphaExt7']
    alphas_df.fillna(method='ffill', inplace=True)
    alphas_df.dropna(inplace=True)
    return alphas_df


def main():
    #import the csv we want to use to calculate alphas 
    data = csv_to_dataframe.csv_to_dataframe('SPY_data_2022_2024.csv')

    alphas_df = alphas_to_df(data)
    print(alphas_to_df(data))

    # #save the alphas to a csv file
    #alphas_df.to_csv('alphas_SPY_2022_2024.csv')

    #print(alphaExt4(data))


if __name__ == "__main__":
    main()
