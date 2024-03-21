import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from functools import reduce
# from sklearn import linear_model

# google = yf.Ticker("GOOG")
# historical_data = google.history(period="5d")

# Print the columns of the dataframe 
# print(historical_data.columns)

# print(historical_data['Open'])


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

    # if(not date == datetime.now().date()):

    #     new_date = x.index[x == date].max()

    #     print("NEW START DATE", new_date)

    #     date = adjust_date(new_date)
    # else:
    #     date = adjust_date(x.index[-1])
    # most_recent_date = date  # Get the most recent date from the index
    # x_days_ago = most_recent_date - timedelta(days=d)
    # print('x_days', x_days_ago)
    # print(x)
    # adj_date = adjust_date(x_days_ago)
    # print('adjusted date', adj_date)
    # col = x.loc[adj_date]  # Use .loc[] to retrieve the row corresponding to the date
    # col = pd.DataFrame(col)
    # return col

    if not isinstance(x, (pd.DataFrame, pd.Series)):
        raise ValueError("x must be a DataFrame or Series")

    if date != datetime.now().date():
        new_date = x.index[x.index == date].max()
        print("NEW START DATE:", new_date)
        date = new_date

    most_recent_date = x.index[-1]  # Get the most recent date from the index
    x_days_ago = date - timedelta(days=d)
    print('x_days:', x_days_ago)
    
    if isinstance(x, pd.DataFrame):
        col = x.loc[x.index == x_days_ago].squeeze()  # Use .loc[] to retrieve the row corresponding to the date
    elif isinstance(x, pd.Series):
        col = x.loc[x.index == x_days_ago]
    
    col = pd.DataFrame(col)
    print('adjusted date:', x_days_ago)
    return col



#time-series max over the past d days 
def ts_max(x, d):
    # d = int(d)
    # max_values = []
    # for i in range(d, len(x)):
    #     window = x[i-d:i]  # Extract the window of size d
    #     max_value = np.max(window)  # Calculate the maximum value in the window
    #     print(max_value)
    #     max_values.append(max_value)
    # return max_values

    max = x.rolling(window=d, min_periods=1).max()
    print("Max: ", max)
    return(max)


 
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
    # ranked_values = x.rolling(window=d).apply(lambda window: pd.Series(window).rank(method='min').iloc[-1])
    # return ranked_values
    # return x.rolling(window=d).apply(lambda x: pd.Series(x).rank(method='first').iloc[-1])
    data = x.tail(d)
    # print('Last d days: ', data)
    data = data.rank(method='first', ascending=False)
    # print('ranked: ', data)
    return data

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
    # print("X is: ", x)
    # x_rank = x.rank(method='first')
    # return x_rank

#time-serial correlation of x and y for the past d days 
def correlation(x, y, d):
    # d = int(d)
    # correlation_vals = []
    # for i in range(len(x) - d + 1):
    #     x_recent = x.iloc[i:i + d]
    #     y_recent = y.iloc[i:i + d]
    #     if not np.isnan(x_recent) and not np.isnan(y_recent):
    #         correlation_vals.append(x_recent.corr(y_recent).any())
    # print("Now printing correlation values:")
    # print(correlation_vals)
    # correlation_vals = pd.Series(correlation_vals, index=x.index[d - 1:])
    # return pd.Series(correlation_vals, index=x.index[d - 1:])
    # d = int(d)
    # # print('x ', x)
    # x_recent = pd.DataFrame(x.tail(d))
    # # print('x_recent: ', x_recent)
    # y_recent = pd.DataFrame(y.tail(d))
    # # print(y_recent)
    
    # # Calculate the correlation
    # correlation = x_recent.corrwith(y_recent, axis=0)
    # # correlation = x_recent.corr(y_recent)
    # # print('correlation: ')
    # # print(correlation)
    # # return correlation
    # d = int(d)
    # correlation = x.rolling(window=d).corr(y)
    # return correlation

    # corr_values = []
    # print('d: ', d)
    # for i in range(d, len(x) + 1):
    #     print("i ", i)
    #     x_window = x.iloc[i - d:i]
    #     print(x_window)
    #     y_window = y.iloc[i - d:i]
    #     print(y_window)
    #     corr_values.append(x_window.corr(y_window))

    # correlations = []
    # print("length is: ", len(x))
    # for i in range(len(x) - d + 1):
    #     print("x: ", x)
    #     x_recent = x.iloc[i:i+d]
    #     print("x_recent: ", x_recent)
    #     y_recent = y.iloc[i:i+d]
    #     correlation = 0
    #     for i in range(0, len(x_recent)):
    #         if x_recent.iloc[i] == y_recent.iloc[i]:
    #             correlation += 1
    #     correlation = correlation / len(x_recent)
    #     correlations.append(correlation)
        # print(corr_values)
    # return pd.Series(corr_values, index=x.index[d - 1:])

    # d = int(d)
    # correlation_vals = []
    # print("lenght be like: ", len(x) - d + 1)
    # for i in range(len(x) - d + 1):
    #     x_recent = x.iloc[i:i + d]
    #     print(x_recent)
    #     y_recent = y.iloc[i:i + d]
    #     print(y_recent)
    #     if not x_recent.isnull().any() and not y_recent.isnull().any():
    #         correlation_vals.append(x_recent.corr(y_recent))
    #         print(x_recent.corr(y_recent))
    # correlation_series = pd.Series(correlation_vals, index=x.index[d - 1:])
    # print("correlation_series: ", correlation_series)
    correlation_series = x.rolling(window=d).corr(y).dropna()
    print("correlation_series: ", correlation_series)
    return correlation_series

    # return correlations


#     x_recent = x[-d:]
#     y_recent = y[-d:]

# # Calculate the cross-correlation
#     correlation = np.correlate(x_recent, y_recent, mode='full')

    # correlation = x.rolling(window=d, center=True).corr(y.dropna(axis=0), pairwise=True).dropna(axis=0)
    # print("Checking le correlation: ")
    # print(correlation)

    # return correlation

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

def product(x, d):
    products = []
    for i in range(len(x)):
        start_idx = max(0, i - d + 1)
        products.append(reduce(lambda a, b: a * b, x[start_idx:i + 1]))
    return products



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

def get_vwap(data):
    # data = yf.download(ticker, period=time_frame, interval="1m")

    # Calculate VWAP
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['CumVol'] = data['Volume'].cumsum()
    data['CumTP'] = (data['TP'] * data['Volume']).cumsum()
    data['VWAP'] = data['CumTP'] / data['CumVol']

    return(data['VWAP'])

def ternary(condition, value_if_true, value_if_false):
    return value_if_true if condition else value_if_false

def log(x):
    return np.log(x)

def sign(x):
    return np.sign(x)

def stddev(x, d):
    std_dev = x.rolling(window=d).std()

    return std_dev



def alpha_26(volume, high):
    all_volume_ranks = ts_rank(volume, 10)
    all_high_ranks = ts_rank(high, 10)
    # print(all_volume_ranks)
    part1 = (ts_rank(volume, 5), ts_rank(high, 5), 5)

    shifted_volumes = []
    shifted_highs = []
    volume_rank = all_volume_ranks.iloc[5:10]
    volume_shifted_1 = all_volume_ranks.iloc[4:9]
    volume_shifted_2 = all_volume_ranks.iloc[3:8]
    volume_shifted_3 = all_volume_ranks.iloc[2:7]
    volume_shifted_4 = all_volume_ranks.iloc[1:6]
    volume_shifted_5 = all_volume_ranks.iloc[0:5]
    shifted_volumes.append(volume_rank)
    shifted_volumes.append(volume_shifted_1)
    shifted_volumes.append(volume_shifted_2)
    shifted_volumes.append(volume_shifted_3)
    shifted_volumes.append(volume_shifted_4)
    shifted_volumes.append(volume_shifted_5)

    high_rank = all_high_ranks.iloc[5:10]
    high_shifted_1 = all_high_ranks.iloc[4:9]
    high_shifted_2 = all_high_ranks.iloc[3:8]
    high_shifted_3 = all_high_ranks.iloc[2:7]
    high_shifted_4 = all_high_ranks.iloc[1:6]
    high_shifted_5 = all_high_ranks.iloc[0:5]
    shifted_highs.append(high_rank)
    shifted_highs.append(high_shifted_1)
    shifted_highs.append(high_shifted_2)
    shifted_highs.append(high_shifted_3)
    shifted_highs.append(high_shifted_4)
    shifted_highs.append(high_shifted_5)

    correlations = []
    for i in range(0, len(shifted_volumes)):
        correlations.append(shifted_volumes[i].corr(shifted_highs[i]))
    print(correlations)

    return -1 * ts_max(correlations, 3)






    # print(volume_shifted_5)


def main():

    data = yf.download('GOOG', period='1y')
    data = pd.DataFrame(data)


    low = data['Low']
    volume = data['Volume'] #THIS REFERS TO VOLUME NOT PRICE VOLUME- should it be changed????
    returns = r(data)   
    close = data['Close']
    high = data['High']
    open = data['Open']
    vwap = get_vwap(data)

    adv20 = adv(data, 20)


    # print("volume: ")
    # print(volume)
    # print('Volume rank: ')
    # print(ts_rank(volume, 5))
    # print('High: ', ts_rank(high, 5))

    starter = ts_rank(volume, 5)
    print('starter: ', starter)


 ####### ALPHA 26 ######
    alpha_26 = (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))

    # print("Getting the correlation: ")
    # print(rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))

### ALPHA 27 ### does not work with current rank function
    # alpha_27 = (ternary((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))), (-1 * 1), 1))

    # print('Printing alpha 26: ', alpha_27)

    # print(alpha_26(volume, high))

### ALPHA 28 ### does not work with current correlation function
    # alpha_28 = scale(((correlation(adv(data, 20), low, 5) + ((high + low) / 2)) - close))

### ALPHA 29 ### the product function product(x, d) must be rewritten to take in all 5 values of x from the beginning
    # alpha_29 = (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))

### ALPHA 30 ###
    alpha_30 = (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))

### ALPHA 31 ### does not work because of correlation
    # alpha_31 = ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))

### ALPHA 32 ### also does not work because of the correlation
    # print("CHECKING CORRELATION: ", correlation(vwap, delay(close, 5), 230))
    # alpha_32 = (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))

### ALPHA 33 ###    
    alpha_33 = rank((-1 * ((1 - (open / close)) ** 1)))

### ALPHA 34 ###
    alpha_34 = rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))

### ALPHA 35 ###
    alpha_35 = ((ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32)))

    print(alpha_35)

    ###### CHECKED UP TO HERE #######
### ALPHA 36 ###
    alpha_36 = (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(ts_rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))

### ALPHA 37 ###
    alpha_37 = (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))

### ALPHA 38 ###
    alpha_38 = ((-1 * rank(ts_rank(close, 10))) * rank((close / open)))

### ALPHA 39 ###
    alpha_39 = ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))

### ALPHA 40 ###
    alpha_40 = ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

### ALPHA 41 ###
    alpha_41 = (((high * low)^0.5) - vwap)

### ALPHA 42 ###
    alpha_42 = (rank((vwap - close)) / rank((vwap + close)))

### ALPHA 43 ###
    alpha_43 = (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))

### ALPHA 44 ###
    alpha_44 = (-1 * correlation(high, rank(volume), 5))

### ALPHA 45 ###
    alpha_45 = (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))

### ALPHA 46 ###
    # alpha_46 = (0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    alpha_46 = ternary((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))), (-1 * 1), ternary(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0), 1, ((-1 * 1) * (close - delay(close, 1)))))

### ALPHA 47 ###
    alpha_47 = ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))

### ALPHA 48 ### do not know how to do industry neutralization
    alpha_48 = (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))

### ALPHA 49 ###
    # alpha_49 = (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    alpha_49 = ternary(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)), 1, ((-1 * 1) * (close - delay(close, 1))))

### ALPHA 50 ###
    alpha_50 = (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))

if __name__ == "__main__":
    main()