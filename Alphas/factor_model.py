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
def combine(func: str, data, w):
    func = getattr(alphas, func)
    df = func(data)
    df.map(lambda x: x * w)
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
    date_time = datetime.datetime.fromtimestamp(int(ts) / 1000)

    return date_time.strftime(strf_fmt)


# Care for race-condition-like edge case. We do not control for way orders are arranged.

# Generate orders

for ticker, data in tqdm(zip(tickers, stock_data)):
    # element-wise linear combination
    calc = pd.DataFrame()
    for func, w in weights.items():
        calc = calc.join(combine(func, data, w), how="outer")

    calc.dropna(inplace=True, axis=0)

    res = calc.agg("sum", axis="columns").apply(lambda x: convert_to_order(x, ticker))
    res.name = ticker

    orders = orders.join(res, how="outer")

orders.dropna(inplace=True, axis=0)
orders.index = orders.index.map(lambda x: unix_to_datetime(x))

cfg.setOrders(orders)

print(orders)
print(f"Done. Took {time.time()-start_time:.2f} seconds.")


# https://python-course.eu/numerical-programming/linear-combinations-in-python.php
