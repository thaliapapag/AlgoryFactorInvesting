import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def adjust_date(date):
    while date.weekday() >= 5:
        date = date + timedelta(days=1)
    
    if date.month == 1 and date.day == 1:
        date = date + timedelta(days =1)
    
    while date.weekday() >= 5:
        date = date + timedelta(days=1)

    # Monday is 0 and Sunday is 6
    return date

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


def get_vwap(data):
    # data = yf.download(ticker, period=time_frame, interval="1m")

    # Calculate VWAP
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['CumVol'] = data['Volume'].cumsum()
    data['CumTP'] = (data['TP'] * data['Volume']).cumsum()
    data['VWAP'] = data['CumTP'] / data['CumVol']

    return(data['VWAP'])



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



def create_row(ticker):

    end_date = datetime.now() - timedelta(days=365)
    start_date = end_date - timedelta(days=365)

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data = pd.DataFrame(data)

    data2 = yf.download(ticker, period='1y')
    data2 = pd.DataFrame(data2)

    low = data['Low']
    volume = data['Volume'] #THIS REFERS TO VOLUME NOT PRICE VOLUME- should it be changed????
    returns = r(data)   
    close = data['Close']
    high = data['High']
    open = data['Open']
    vwap = get_vwap(data)

    alpha_1 = np.mean(alpha1(data))
    # print("alpha_1: ", alpha_1)
    alpha_2 = np.mean(alpha2(data))
    # print("alpha_2: ", alpha_2)
    alpha_3 = np.mean(alpha3(data))
    # print("alpha_3: ", alpha_3)
    alpha_4 = np.mean(alpha4(data))
    # print("alpha_4: ", alpha_4)
    alpha_5 = np.mean(alpha5(data))
    # print("alpha_5: ", alpha_5)
    alpha_6 = np.mean(alpha6(data))
    # print("alpha_6: ", alpha_6)
    # alpha_7 = np.mean(alpha7(data)) Kept giving me nan values as answer
    # print("alpha_7: ", alpha_7)
    # alpha_8 = alpha8(data) Wasn't working in the moment
    # alpha_9 = np.mean(alpha9(data)) Kept giving me nan values as answer
    # print("alpha_9: ", alpha_9)
    alpha_41 = np.mean((((high * low) ** 0.5) - vwap))
    # print("alpha_41: ", alpha_41)
    # print("printed alpha")

    # alpha_51 = alpha51(close) Wasn't working in the moment
    alpha_52 = np.mean(alpha52(low, returns, volume))
    # print("alpha_52: ", alpha_52)
    alpha_53 = np.mean(alpha53(close, low, high))
    # print("alpha_53: ", alpha_53)
    alpha_54 = np.mean(alpha54(low, close, open, high))
    # print("alpha_54: ", alpha_54)
    # alpha_55 = alpha55(close, low, high, volume) Removed because was returning an empty dataframe
    # print("alpha_55: ", alpha_55)

    price_change = (data2['Close'].iloc[6] - data2['Close'].iloc[0]) / data2['Close'].iloc[0] * 100 # Gives a percentage price change

    # change the first .iloc[xx] value here to alter the time frame of price change: [6] is for a week, [-1] is for a year (the most recent value)

    new_row = [ticker, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_41, alpha_52, alpha_53, alpha_54, price_change]

    return new_row








def main():

# Use these lines to get stock data between 2 and 1 years ago (used for training as opposed to data from the last year)
    end_date = datetime.now() - timedelta(days=365)
    start_date = end_date - timedelta(days=365)

    # data = yf.download('GOOG', period='1y')

    ticker = 'GOOG'

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data = pd.DataFrame(data)

    data2 = yf.download(ticker, period='1y')
    data = pd.DataFrame(data2)

    # row1 = create_row('GOOG')
    # row2 = create_row('AAPL')
    # row3 = create_row('MSFT')
    # row4 = create_row('DPZ')
    # row5 = create_row('MIDD')
    # row6 = create_row('AMZN')
    # row7 = create_row('TSLA')
    # row8 = create_row('NFLX')
    # row9 = create_row('META')
    # row10 = create_row('NVDA')
    # row11 = create_row('JPM')
    # row12 = create_row('BAC')
    # row13 = create_row('WFC')
    # row14 = create_row('C')
    # row15 = create_row('GS')
    # row16 = create_row('PYPL')
    # row17 = create_row('SQ')
    # row18 = create_row('V')
    # row19 = create_row('MA')
    # row20 = create_row('AXP')
    # row21 = create_row('T')
    # row22 = create_row('VZ')
    # row23 = create_row('TMUS')
    # row24 = create_row('CMCSA')
    # row25 = create_row('DIS')

    stock_tickers = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "AMZN",  # Amazon.com Inc.
    "GOOGL", # Alphabet Inc. (Google)
    "META",    # Meta Platforms Inc. (Facebook)
    "TSLA",  # Tesla Inc.
    "JPM",   # JPMorgan Chase & Co.
    "V",     # Visa Inc.
    "PG",    # Procter & Gamble Company
    "NVDA",  # NVIDIA Corporation
    "MA",    # Mastercard Incorporated
    "DIS",   # The Walt Disney Company
    "HD",    # The Home Depot Inc.
    "PYPL",  # PayPal Holdings Inc.
    "NFLX",  # Netflix Inc.
    "CMCSA", # Comcast Corporation
    "INTC",  # Intel Corporation
    "CSCO",  # Cisco Systems Inc.
    "PEP",   # PepsiCo Inc.
    "ADBE",  # Adobe Inc.
    "CRM",   # Salesforce.com Inc.
    "BAC",   # Bank of America Corporation
    "KO",    # The Coca-Cola Company
    "WMT",   # Walmart Inc.
    "IBM",   # International Business Machines Corporation
    "XOM",   # Exxon Mobil Corporation
    "MRK",   # Merck & Co. Inc.
    "NKE",   # NIKE Inc.
    "PFE",   # Pfizer Inc.
    "ABBV",  # AbbVie Inc.
    "MO",    # Altria Group Inc.
    "ORCL",  # Oracle Corporation
    "T",     # AT&T Inc.
    "UNH",   # UnitedHealth Group Incorporated
    "CVX",   # Chevron Corporation
    "VZ",    # Verizon Communications Inc.
    "TMO",   # Thermo Fisher Scientific Inc.
    "COST",  # Costco Wholesale Corporation
    "ABT",   # Abbott Laboratories
    "AMGN",  # Amgen Inc.
    "LMT",   # Lockheed Martin Corporation
    "AVGO",  # Broadcom Inc.
    "TXN",   # Texas Instruments Incorporated
    "HON",   # Honeywell International Inc.
    "NEE",   # NextEra Energy Inc.
    "DHR",   # Danaher Corporation
    "SBUX",  # Starbucks Corporation
    "UPS",   # United Parcel Service Inc.
    "BA",    # The Boeing Company
    "LLY",   # Eli Lilly and Company
    "AMD",   # Advanced Micro Devices Inc.
    "MCD",   # McDonald's Corporation
    "MMM",   # 3M Company
    "GS",    # The Goldman Sachs Group Inc.
    "CAT",   # Caterpillar Inc.
    "FDX",   # FedEx Corporation
    "CVS",   # CVS Health Corporation
    "TGT",   # Target Corporation
    "BKNG",  # Booking Holdings Inc.
    "CI",    # Cigna Corporation
    "BDX",   # Becton, Dickinson and Company
    "MS",    # Morgan Stanley
    "LOW",   # Lowe's Companies Inc.
    "GE",    # General Electric Company
    "F",     # Ford Motor Company
    "DUK",   # Duke Energy Corporation
    "SO",    # The Southern Company
    "C",     # Citigroup Inc.
    "AXP",   # American Express Company
    "GM",    # General Motors Company
    "USB",   # U.S. Bancorp
    "INTU",  # Intuit Inc.
    "AMT",   # American Tower Corporation
    "WFC",   # Wells Fargo & Company
    "RTX",   # Raytheon Technologies Corporation
    "BLK",   # BlackRock Inc.
    "SPG",   # Simon Property Group Inc.
    "CL",    # Colgate-Palmolive Company
    "PLD",   # Prologis Inc.
    "MET",   # MetLife Inc.
    "MMC",   # Marsh & McLennan Companies Inc.
    "MDT",   # Medtronic plc
    "ZTS",   # Zoetis Inc.
    "SO",    # The Southern Company
    "KMB",   # Kimberly-Clark Corporation
    "EQIX",  # Equinix Inc.
    "REGN",  # Regeneron Pharmaceuticals Inc.
    "GD",    # General Dynamics Corporation
    "ICE",   # Intercontinental Exchange Inc.
    "CSX",   # CSX Corporation
    "EMR",   # Emerson Electric Co.
    "CCI",   # Crown Castle International Corp.
    "VRTX",  # Vertex Pharmaceuticals Incorporated
    "APD",   # Air Products and Chemicals Inc.
    "ADI",   # Analog Devices Inc.
    "APTV",  # Aptiv PLC
    "TJX",   # The TJX Companies Inc.
    "AEP",   # American Electric Power Company Inc.
    "NOC",   # Northrop Grumman Corporation
    "ES",    # Eversource Energy
    "PSX",   # Phillips 66
    "MMC",   # Marsh & McLennan Companies Inc.
    "NEM",   # Newmont Corporation
    "EW",    # Edwards Lifesciences Corporation
    "ETN",   # Eaton Corporation plc
    "HUM",   # Humana Inc.
    "ECL",   # Ecolab Inc.
    "AMAT",  # Applied Materials Inc.
    "EL",    # The Estée Lauder Companies Inc.
    "SRE",   # Sempra Energy
    "PEG",   # Public Service Enterprise Group Incorporated
    "FIS",   # Fidelity National Information Services Inc.
    "AON",   # Aon plc
    "EBAY",  # eBay Inc.
    "CME",   # CME Group Inc.
    "MCO",   # Moody's Corporation
    "BMY",   # Bristol-Myers Squibb Company
    "SYY",   # Sysco Corporation
    "WBA",   # Walgreens Boots Alliance Inc.
    "DXCM",  # DexCom Inc.
    "WELL",  # Welltower Inc.
    "ORLY",  # O'Reilly Automotive Inc.
    "LHX",   # L3Harris Technologies Inc.
    "LIN",   # Linde plc
    "D",     # Dominion Energy Inc.
    "FTV",   # Fortive Corporation
    "ROST",  # Ross Stores Inc.
    "GD",    # General Dynamics Corporation
    "SCHW",  # The Charles Schwab Corporation
    "CMS",   # CMS Energy Corporation
    "A",     # Agilent Technologies Inc.
    "KEYS",  # Keysight Technologies Inc.
    "DG",    # Dollar General Corporation
    "QRVO",  # Qorvo Inc.
    "RMD",   # ResMed Inc.
    "LUV",   # Southwest Airlines Co.
    "TT",    # Trane Technologies plc
    "PEG",   # Public Service Enterprise Group Incorporated
    "BK",    # The Bank of New York Mellon Corporation
    "MTD",   # Mettler-Toledo International Inc.
    "EFX",   # Equifax Inc.
    "PGR",   # The Progressive Corporation
    "IDXX",  # IDEXX Laboratories Inc.
    "MCK",   # McKesson Corporation
    "KMI",   # Kinder Morgan Inc.
    "TYL",   # Tyler Technologies Inc.
    "HCA",   # HCA Healthcare Inc.
    "IQV",   # IQVIA Holdings Inc.
    "VFC",   # V.F. Corporation
    "RSG",   # Republic Services Inc.
    "VRSK",  # Verisk Analytics Inc.
    "PAYX",  # Paychex Inc.
    "PH",    # Parker-Hannifin Corporation
    "PEAK",  # Healthpeak Properties Inc.
    "CTSH",  # Cognizant Technology Solutions Corporation
    "NTRS",  # Northern Trust Corporation
    "AKAM",  # Akamai Technologies Inc.
    "AWK",   # American Water Works Company Inc.
    "ESS",   # Essex Property Trust Inc.
    "ULTA",  # Ulta Beauty Inc.
    "OTIS",  # Otis Worldwide Corporation
    "MNST",  # Monster Beverage Corporation
    "CTLT",  # Catalent Inc.
    "TTWO",  # Take-Two Interactive Software Inc.
    "PAYC",  # Paycom Software Inc.
    "WST",   # West Pharmaceutical Services Inc.
    "CARR",  # Carrier Global Corporation
    "TFX",   # Teleflex Incorporated
    "CDNS",  # Cadence Design Systems Inc.
    "DRI",   # Darden Restaurants Inc.
    "DTE",   # DTE Energy Company
    "ATO",   # Atmos Energy Corporation
    "TDY",   # Teledyne Technologies Incorporated
    "LH",    # Laboratory Corporation of America Holdings
    "GWW",   # W.W. Grainger Inc.
    "CBRE",  # CBRE Group Inc.
    "FLT",   # FleetCor Technologies Inc.
    "HII",   # Huntington Ingalls Industries Inc.
    "KHC",   # The Kraft Heinz Company
    "CDW",   # CDW Corporation
    "ROL",   # Rollins Inc.
    "LRCX",  # Lam Research Corporation
    "WAT",   # Waters Corporation
    "TYL",   # Tyler Technologies Inc.
    "FFIV",  # F5 Networks Inc.
    "MSI",   # Motorola Solutions Inc.
    "WDC",   # Western Digital Corporation
    "CMI",   # Cummins Inc.
    "K",     # Kellogg Company
    "RHI",   # Robert Half International Inc.
    "KEY",   # KeyCorp
    "HES",   # Hess Corporation
    "AKZOY", # Akzo Nobel N.V.
    "BAX",   # Baxter International Inc.
    "CINF",  # Cincinnati Financial Corporation
    "ANSS",  # ANSYS Inc.
    "CBOE",  # Cboe Global Markets Inc.
    "PXD",   # Pioneer Natural Resources Company
    "CTVA",  # Corteva Inc.
    "PKG",   # Packaging Corporation of America
    "HSY",   # The Hershey Company
    "SNPS",  # Synopsys Inc.
    "COO",   # The Cooper Companies Inc.
    "RF",    # Regions Financial Corporation
    "AEE",   # Ameren Corporation
    "ATO",   # Atmos Energy Corporation
    "ROK",   # Rockwell Automation Inc.
    "DXC",   # DXC Technology Company
    "AME",   # AMETEK Inc.
    "AZO",   # AutoZone Inc.
    "LDOS",  # Leidos Holdings Inc.
    "PEAK",  # Healthpeak Properties Inc.
    "JNPR",  # Juniper Networks Inc.
    "MLM",   # Martin Marietta Materials Inc.
    "MGM",   # MGM Resorts International
    "WY",    # Weyerhaeuser Company
    "CMS",   # CMS Energy Corporation
    "NTRS",  # Northern Trust Corporation
    "ETSY",  # Etsy Inc.
    "FLT",   # FleetCor Technologies Inc.
    "ODFL",  # Old Dominion Freight Line Inc.
    "AVB",   # AvalonBay Communities Inc.
    "AIZ",   # Assurant Inc.
    "WRK",   # WestRock Company
    "FTI",   # TechnipFMC plc
    "EXPD",  # Expeditors International of Washington Inc.
    "GL",    # Globe Life Inc.
    "HST",   # Host Hotels & Resorts Inc.
    "URI",   # United Rentals Inc.
    "NVR",   # NVR Inc.
    "RCL",   # Royal Caribbean Group
    "LKQ",   # LKQ Corporation
    "WAB",   # Westinghouse Air Brake Technologies Corporation
    "WHR",   # Whirlpool Corporation
    "PVH",   # PVH Corp.
    "MLM",   # Martin Marietta Materials Inc.
    "LKQ",   # LKQ Corporation
    "AZO",   # AutoZone Inc.
    "MOS",   # The Mosaic Company
    "HBI",   # Hanesbrands Inc.
    "MKC",   # McCormick & Company Incorporated
    "AMP",   # Ameriprise Financial Inc.
    "HOLX",  # Hologic Inc.
    "TPR",   # Tapestry Inc.
    "VMC",   # Vulcan Materials Company
    "JBHT",  # J.B. Hunt Transport Services Inc.
    "LEG",   # Leggett & Platt Incorporated
    "WU",    # Western Union Company
    "IPGP",  # IPG Photonics Corporation
    "COTY",  # Coty Inc.
    "HAS",   # Hasbro Inc.
    "NUE",   # Nucor Corporation
    "BKR",   # Baker Hughes Company
    "HPE",   # Hewlett Packard Enterprise Company
    "VTRS",  # Viatris Inc.
    "KEYS",  # Keysight Technologies Inc.
    "GLW",   # Corning Incorporated
    "OKE",   # ONEOK Inc.
    "IT",    # Gartner Inc.
    "FE",    # FirstEnergy Corp.
    "PNW",   # Pinnacle West Capital Corporation
    "GRMN",  # Garmin Ltd.
    "SWK",   # Stanley Black & Decker Inc.
    "CNC",   # Centene Corporation
    "PKG",   # Packaging Corporation of America
    "GPN",   # Global Payments Inc.
    "TDG",   # TransDigm Group Incorporated
    "BIO",   # Bio-Rad Laboratories Inc.
    "ARE",   # Alexandria Real Estate Equities Inc.
    "ANET",  # Arista Networks Inc.
    "IPG",   # The Interpublic Group of Companies Inc.
    "TSCO",  # Tractor Supply Company
    "LDOS",  # Leidos Holdings Inc.
    "ALLE",  # Allegion plc
    "KMX",   # CarMax Inc.
    "MHK",   # Mohawk Industries Inc.
    "CAG",   # Conagra Brands Inc.
    "WYNN",  # Wynn Resorts Limited
    "EXR",   # Extra Space Storage Inc.
    "KMI",   # Kinder Morgan Inc.
    "AVY",   # Avery Dennison Corporation
    "HII",   # Huntington Ingalls Industries Inc.
    "TDG",   # TransDigm Group Incorporated
    "CPRT",  # Copart Inc.
    "NDAQ",  # Nasdaq Inc.
    "IP",    # International Paper Company
    "FRT",   # Federal Realty Investment Trust
    "VNO",   # Vornado Realty Trust
    "HIG",   # Hartford Financial Services Group Inc.
    "RMD",   # ResMed Inc.
    "CNP",   # CenterPoint Energy Inc.
    "SNV",   # Synovus Financial Corp.
    "PKG",   # Packaging Corporation of America
    "HBI",   # Hanesbrands Inc.
    "FFIV",  # F5 Networks Inc.
    "PHM",   # PulteGroup Inc.
    "ALK",   # Alaska Air Group Inc.
    "LNT",   # Alliant Energy Corporation
    "MTB",   # M&T Bank Corporation
    "QRVO",  # Qorvo Inc.
    "LVS",   # Las Vegas Sands Corp.
    "NTAP",  # NetApp Inc.
    "ALLE",  # Allegion plc
    "AOS",   # A. O. Smith Corporation
    "HII",   # Huntington Ingalls Industries Inc.
    "HOLX",  # Hologic Inc.
    "PNR",   # Pentair plc
    "EXPD",  # Expeditors International of Washington Inc.
    "KEY",   # KeyCorp
    "DHI",   # D.R. Horton Inc.
    "NUE",   # Nucor Corporation
    "MAS",   # Masco Corporation
    "LNC",   # Lincoln National Corporation
    "BR",    # Broadridge Financial Solutions Inc.
    "RHI",   # Robert Half International Inc.
    "J",     # Jacobs Engineering Group Inc.
    "HSIC",  # Henry Schein Inc.
    "NWSA",  # News Corporation
    "FMC",   # FMC Corporation
    "REG",   # Regency Centers Corporation
    "WRB",   # W. R. Berkley Corporation
    "GILD",  # Gilead Sciences Inc.
    "INCY",  # Incyte Corporation
    "PNW",   # Pinnacle West Capital Corporation
    "JCI",   # Johnson Controls International plc
    "ROL",   # Rollins Inc.
    "WHR",   # Whirlpool Corporation
    "VNO",   # Vornado Realty Trust
    "TSCO",  # Tractor Supply Company
    "CNP",   # CenterPoint Energy Inc.
    "VTR",   # Ventas Inc.
    "OTIS",  # Otis Worldwide Corporation
    "SNV",   # Synovus Financial Corp.
    "CINF",  # Cincinnati Financial Corporation
    "IRM",   # Iron Mountain Incorporated
    "AJG",   # Arthur J. Gallagher & Co.
    "ZBRA",  # Zebra Technologies Corporation
    "HST",   # Host Hotels & Resorts Inc.
    "ROL",   # Rollins Inc.
    "CLX",   # The Clorox Company
    ]

############### CREATE THE ROWS AND TURN THEM INTO A CSV. THIS IS WHERE TO MAKE A NEW DATASET FOR USE LATER ####################################

    # rows = {}
    # for ticker in stock_tickers:
    #     rows[ticker] = create_row(ticker)

    # print("printing the ticker list with alphas: ")
    # print(rows)


    # dataframe = pd.DataFrame.from_dict(rows, orient='index', columns=['ticker', 'alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'alpha_5', 'alpha_6', 'alpha_41', 'alpha_52', 'alpha_53', 'alpha_54', 'price_change'])
    # dataframe.to_csv('data.csv', index=False)




    

    # print("COMPLETED MAIN task")
    # print("Row 1: ", row1)
    # print("Row 2: ", row2)
    # print("Row 3: ", row3)
    # print("Row 4: ", row4)
    # print("Row 5: ", row5)
    

    # df = pd.DataFrame([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15, row16, row17, row18, row19, row20, row21, row22, row23, row24, row25])
    # df.columns = ['ticker', 'alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'alpha_5', 'alpha_6', 'alpha_41', 'alpha_52', 'alpha_53', 'alpha_54', 'price_change']
    
    df = pd.read_csv('data.csv')

    print(df)
    scaler = StandardScaler()

    x = df.iloc[:, 1:-1]
    nan_columns = df.columns[df.isna().any()].tolist()
    print(nan_columns)
    print("Infinity values:", np.any(np.isinf(x)))
    print("Max value:", np.max(x))
    x[np.isinf(x)] = np.finfo(np.float64).max
    x = scaler.fit_transform(x)
    print(x)
    y = df.iloc[:, -1]
    # y = scaler.fit_transform(y.values.reshape(-1, 1))
    print(y)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    regressor = DecisionTreeRegressor()
    regressor.fit(xTrain, yTrain)

    yPred = regressor.predict(xTest)

    mse = mean_squared_error(yTest, yPred) # These values indicated that the decision tree regressor did a pretty poor job
    print("Mean Squared Error (Decision tree):", mse)

    r_squared = r2_score(yTest, yPred)
    print("R-squared (Decision tree):", r_squared)


    model = LinearRegression()
    model.fit(xTrain, yTrain)

    y_pred = model.predict(xTest)

    mse = mean_squared_error(yTest, y_pred)
    r2 = r2_score(yTest, y_pred)

    print(f"Mean Squared Error (Linear Regression): {mse}") # Linear regression did even worse
    print(f"R-squared (Linear Regression): {r2}")


    forest = RandomForestRegressor(n_estimators=400, random_state=42) # The random forest is most accurate but leaves room for improvement

    forest.fit(xTrain, yTrain)
 
    y_pred = forest.predict(xTest)

    mse = mean_squared_error(yTest, y_pred)
    r2 = r2_score(yTest, y_pred)

    print(f"Mean Squared Error (random forest): {mse}")
    print(f"R-squared (random forest): {r2}")

    # print("GETTING DATA: ")
    # print(data)

#### The neural network; could be the best bet.
    model = Sequential([
    Dense(64, activation='relu', input_shape=(xTrain.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer with 1 neuron for regression
])
    

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    model.fit(xTrain, yTrain, epochs=10, batch_size=32, validation_data=(xTest, yTest))

    y_pred = model.predict(xTest)
    print("Predicted Results: ")
    print(y_pred)
    print("Actual Results: ")
    print(yTest)

    loss, mae = model.evaluate(xTest, yTest)
    print("Mean Absolute Error for NN:", mae)






    low = data['Low']
    volume = data['Volume'] #THIS REFERS TO VOLUME NOT PRICE VOLUME- should it be changed????
    returns = r(data)   
    close = data['Close']
    high = data['High']
    open = data['Open']
    vwap = get_vwap(data)

    adv20 = adv(data, 20)

    ####### ALPHA 26 ###### COUNT IT
    alpha_26 = (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    

    ### ALPHA 35 ### this may actually work well, COUNT IT
    # print(ts_rank(volume, 32))
    alpha_35 = ((ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32)))
    

    # ### ALPHA 41 ### this one works!
    alpha_41 = (((high * low) ** 0.5) - vwap)
    # print(alpha_41)

if __name__ == "__main__":
    main()