import yfinance as yf
from datetime import timedelta
from datetime import datetime
import numpy as np
from sklearn import linear_model
import pandas as pd

#methods - Thalia

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
    print('x_days', x_days_ago)
    print(x)
    adj_date = adjust_date(x_days_ago)
    print('adjusted date', adj_date)
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
    d = int(d)
    x_recent = x.tail(d)
    y_recent = y.tail(d)
    
    # Calculate the correlation
    correlation = x_recent.corr(y_recent)
    
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
#print(((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
def alpha51(close):
    if (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05):
        alpha51= 1 
    else : 
        alpha51 = ((-1 * 1) * (close - delay(close, 1)))
    return alpha51

#Alpha#52: works
def alpha52(low, returns, volume):
    alpha52 = ((((-1 * ts_min(low, 5)) + delay(low, 5, ts_min(low, 5))) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    return alpha52

#Alpha#53: works 
def alpha53(close, low, high):
    alpha53 = (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    return alpha53

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

    print(alpha51(close))
    print(alpha52(low, returns, volume))
    print(alpha53(close, low, high))
    


if __name__ == "__main__":
    main()
