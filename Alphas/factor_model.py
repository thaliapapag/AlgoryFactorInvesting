import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import json
import datetime
from config import cfg
from data_download import get_spy_tkr_data
import ALPHAS_FINAL as alphas
from csv_to_dataframe import create_y
from ALPHAS_FINAL import alphas_to_df

"""
To-Do:
Slice-dates (generate orders in range of specific dates, preferably by dates->index)
    - Cast dates to indices
"""

print(f"Working directory: {os.getcwd()}")

root = "Alphas"

# See config.py and config.json for CONFIG SETTINGS

THRESHOLD_BUY = cfg.opt["threshold_buy"]
THRESHOLD_SELL = cfg.opt["threshold_sell"]
ORDER_SIZE = cfg.opt["order_size"]  # can also do floor div of $ amt by Price/Share

# data = pd.read_csv(os.path.join(root, "Data/spy.csv"))
# if we want to backtest on all tickers we can push data.columns
tickers = np.loadtxt(os.path.join(root, "Data/spy_tickers.txt"), dtype="str")
stock_data = []
weights = cfg.weights  # key,value : alpha (function name),weight. Ex: alpha1 : 0.58
day_range = cfg.opt["day_range"]  # tmp spaghetti
orders = pd.DataFrame()

start_time = time.time()

for ticker in tqdm(tickers):
    tkr_path = os.path.join(root, f"Data/Stock_History/{ticker}_info.json")
    if not os.path.exists(tkr_path):
        get_spy_tkr_data(ticker=ticker)

    with open(tkr_path) as f:
        data = json.loads(json.load(f))

    data = pd.DataFrame.from_dict(data)
    stock_data.append(data)

# sliced_data = data.iloc[-50:].to_numpy()


# assumes that func only has data as arg (or as *args)
def combine_series(func_str: str, data, w):
    func = getattr(alphas, func_str)
    df = func(data)
    df.name = func_str
    df.map(lambda x: x * w)
    return df


def combine_df(func_str: str, data, weights):
    # returns dataframe of all alphas
    func = getattr(alphas, func_str)
    df = func(data)

    for alpha, w in weights.items():
        df[alpha] = df[alpha].map(lambda x: w)

    return df


# order_type, symbol, quantity = order
def convert_to_order(num, ticker):
    if num > THRESHOLD_BUY:
        return ["BUY", ticker, ORDER_SIZE]
    elif num < THRESHOLD_SELL:
        return ["SELL", ticker, ORDER_SIZE]

    return ["HOLD", ticker, ORDER_SIZE]


def unix_to_datetime(ts: int, strf_fmt=cfg.fmt):
    """
    Takes unix timestep (which we have in json) and converts it into datetime object.
    @ts: unix timestamp, represented in miliseconds
    """
    try:
        date_time = datetime.datetime.fromtimestamp(int(ts) / 1000)
        return date_time.strftime(strf_fmt)
    except Exception as e:
        print("Error in unix_to_datetime: ", e)


# Care for race-condition-like edge case. We do not control for way orders are arranged.

# Generate orders

for ticker, data in tqdm(zip(tickers, stock_data)):
    # element-wise linear combination
    calc = pd.DataFrame()
    # for func, w in weights.items():
    #     print(data.index)
    #     data.index = data.index.map(lambda x: unix_to_datetime(x))
    #     calc = calc.join(combine_series(func, data, w), how="outer")
    #    y.dropna(inplace=True, axis=0)
    #     res = y.agg("sum", axis="columns").apply(lambda x: convert_to_order(x, ticker))

    # JKLJF:DA
    # data.index = data.index.map(lambda x: unix_to_datetime(x))
    # calc = alphas_to_df(data)

    # print(calc)

    # y, calcf = create_y(data, calc)

    # print(y, data)

    # print(calcf)

    # res = y.apply(lambda x: convert_to_order(x, ticker))
    # print(res)
    # res.name = ticker

    # orders = orders.join(res, how="outer")
    pass

for ticker in ["AAPL", "AMZN", "DIS", "GOOGL", "MSFT", "NFLX", "RL", "TSLA"]:
    df = pd.read_csv(os.path.join(root, f"stock_x_y/{ticker}_x.csv")).set_index("Date")

    for alpha in df.columns:
        weight = weights[alpha]
        df[alpha] = df[alpha] * weight

    print(df.agg("sum", axis="columns"))

    res = df.agg("sum", axis="columns").apply(lambda x: convert_to_order(x, ticker))
    print(res)
    res.name = ticker

orders = orders.join(res, how="outer")
orders.dropna(inplace=True, axis=0)
# orders.index = orders.index.map(lambda x: unix_to_datetime(x))

cfg.setOrders(orders)

print(orders)
print(f"Done. Took {time.time()-start_time:.2f} seconds.")


# https://python-course.eu/numerical-programming/linear-combinations-in-python.php
