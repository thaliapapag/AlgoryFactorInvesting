import yfinance as yf
from datetime import timedelta
from datetime import datetime
import numpy as np
from sklearn import linear_model
import pandas as pd


#For alpha equations need:
'''
returns = daily close-to-close returns 
open, close, high, low, volume = standard definitions for daily price and volume data 
vwap = daily volume-weighted average price 
cap = market cap  
adv{d} = average daily dollar volume for the past d days 
IndClass = a generic placeholder for a binary industry classification such as GICS, BICS, NAICS, 
SIC, etc., in indneutralize(x, IndClass.level), where level = sector, industry, subindustry, etc.  
Multiple IndClass in the same alpha need not correspond to the same industry classification.
'''
#DEFINITIONS
'''
(Below “{ }” stands for a placeholder.  All expressions are case insensitive.) 
abs(x), log(x), sign(x) = standard definitions; same for the operators “+”, “-”, “*”, “/”, “>”, “<”, 
“==”, “||”, “x ? y : z” 
rank(x) = cross-sectional rank 
delay(x, d) = value of x d days ago  
correlation(x, y, d) = time-serial correlation of x and y for the past d days 
covariance(x, y, d) = time-serial covariance of x and y for the past d days 
scale(x, a) = rescaled x such that sum(abs(x)) = a (the default is a = 1) 
delta(x, d) = today’s value of x minus the value of x d days ago 
signedpower(x, a) = x^a 
decay_linear(x, d) = weighted moving average over the past d days with linearly decaying 
weights d, d – 1, ..., 1 (rescaled to sum up to 1) 
indneutralize(x, g) = x cross-sectionally neutralized against groups g (subindustries, industries, 
sectors, etc.), i.e., x is cross-sectionally demeaned within each group g 
ts_{O}(x, d) = operator O applied across the time-series for the past d days; non-integer number 
of days d is converted to floor(d)  
ts_min(x, d) = time-series min over the past d days 
16 
 
ts_max(x, d) = time-series max over the past d days 
ts_argmax(x, d) = which day ts_max(x, d) occurred on 
ts_argmin(x, d) = which day ts_min(x, d) occurred on 
ts_rank(x, d) = time-series rank in the past d days 
min(x, d) = ts_min(x, d) 
max(x, d) = ts_max(x, d) 
sum(x, d) = time-series sum over the past d days 
product(x, d) = time-series product over the past d days 
stddev(x, d) = moving time-series standard deviation over the past d days
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


    #
    # print(adv(data, 180))
    
    #print(adv(data, 5))

    #print("TS", ts_min(low, 5))

    #print(delay(close, 3))

    #Alpha#51: 
    #TODO: THIS COMES BACK AS AN ARRAY- AVERAGE AND SUM?
    #print(((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
    '''
    if (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05):
        alpha51= 1 
    else : 
        alpha51 = ((-1 * 1) * (close - delay(close, 1)))

    print(alpha51)
    '''
    #Alpha#52: works
    '''
    alpha52 = ((((-1 * ts_min(low, 5)) + delay(low, 5, ts_min(low, 5))) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    
    print(alpha52)
    '''

    #Alpha#53: works 
    '''
    alpha53 = (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    print(alpha53)
    '''
    #Alpha#54: works 
    # ^ sign replaced with ** 
    '''
    alpha54 = ((-1 * ((low - close) * (open**5))) / ((low - high) * (close**5))) 
    print(alpha54)
    '''

    #Alpha#55: 
    
    #print((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))
    #print(rank(volume))
    '''
    alpha55 = (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    print(alpha55)
    '''
    #Alpha#56: UNABLE TO GET MARKET CAP 
    '''
    alpha56 = (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    '''
    #TODO
    #Alpha#57: 
    '''
    print(ts_argmax(close, 30))
    print(rank(ts_argmax(close, 30)), 2)
    print(decay_linear(rank(ts_argmax(close, 30)), 2))
    print((close - vwap))

    # Calculate decay_linear(rank(ts_argmax(close, 30)), 2) to get the decay array
    decay_array = decay_linear(rank(ts_argmax(close, 30)), 2)

    # Ensure that decay_array has the same length as the close DataFrame
    decay_array = decay_array[-len(close):]

    # Perform element-wise division between close and vwap
    result = (close - vwap) / decay_array
    
    alpha57 = (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))) 
    '''

    '''
    #Alpha#58: DONT HAVE INDUSTRY DATA
    alpha58 = (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
    '''
    
    '''
    #Alpha#59: 
    alpha59 = (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    '''

    
    #Alpha#60: error, divide by zero
    '''
    alpha60 = (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10)))))) 
    print(alpha60)
    '''
    
    #Alpha#61: correlation 
    #print(vwap - ts_min(vwap, 16.1219))
    '''
    a = adv(data, 180)
    adv180 = np.full((1, 18), a)
    adv_df = pd.DataFrame(adv180)
    
    alpha61 = (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv_df, 17.9282)))
    print(alpha61)
    '''
    
    #Alpha#62: numy float object is 
    ''' 
    alpha62 = ((rank(correlation(vwap, sum(adv(data, 20), 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    print(alpha62)
    '''
    
    
    #Alpha#63: requires industry
    ''' 
    alpha63 = ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv(data, 180), 37.2467), 13.557), 12.2883))) * -1)
    print(alpha63)
    '''
    
    #Alpha#64:TODO
    '''
    alpha64 = ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv(data, 120), 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    print(alpha64)
    '''

    '''
    #Alpha#65: 
    alpha65 = ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv(data, 60), 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    '''

    '''
    #Alpha#66: 
    alpha66 = ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    '''
    
    '''
    #Alpha#67: 
    alpha67 = ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv(data, 20), IndClass.subindustry), 6.02936))) * -1)
    '''
    
    '''
    #Alpha#68: 
    alpha68 = ((Ts_Rank(correlation(rank(high), rank(adv(data, 15)), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    '''
    
    '''
    #Alpha#69: 
    alpha69 = ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv(data, 20), 4.92416), 9.0615)) * -1) 
    '''
    
    '''
    #Alpha#70: 
    alpha70 = ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv(data, 50), 17.8256), 17.9171)) * -1) 
    '''
    
    '''
    #Alpha#71: 
    alpha71 = max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv(data, 180), 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
    '''
    
    '''
    #Alpha#72: 
    alpha72 = (rank(decay_linear(correlation(((high + low) / 2), adv(data, 40), 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011))) 
    '''
    
    '''
    #Alpha#73: 
    alpha73 = (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    '''
    
    '''
    #Alpha#74: 
    alpha74 = ((rank(correlation(close, sum(`   `(data, 30), 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1) 
    '''
    
    
    #Alpha#75: 
    '''
    alpha75 = (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv(data, 50)), 12.4413)))
    '''

if __name__ == "__main__":
    main()

