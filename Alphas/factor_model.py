import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import json
from data_download import get_spy_tkr_data
from market_tester import run_timeline

"""
To-Do:
Slice-dates (generate orders in range of specific dates, preferably by dates->index)
    - Cast dates to indices
Sync with alpha functions (alpha parameters need to be standardized for this to work cleanly)

"""

print(f"Working directory: {os.getcwd()}")


class Config:
    def __init__(self, path="Data/Config", root=True):
        self.config_root = os.path.join(root, path) if root else path
        with open(os.path.join(self.config_root, "config.json")):
            self.cfg = json.loads(f.read())
            self.weights = self.cfg["weights"]
            self.opt = self.cfg["options"]


### CONFIG ###

# Modify Config/config.json to update parameters for Grid Search
cfg = Config()

### CONFIG ###
THRESHOLD_BUY = cfg.opt["threshold_buy"]
THRESHOLD_SELL = cfg.opt["threshold_sell"]
ORDER_SIZE = cfg.opt["order_size"]  # can also do floor div of $ amt by Price/Share
root = "Alphas"

# data = pd.read_csv(os.path.join(root, "Data/spy.csv"))
# if we want to backtest on all tickers we can push data.columns
tickers = np.loadtxt(os.path.join(root, "Data/spy_tickers.txt"), dtype="str")
stock_data = []
weights = cfg.weights  # key,value : alpha (function name),weight. Ex: alpha1 : 0.58
day_range = cfg.opt["day_range"]  # tmp spaghetti
orders = [] * day_range

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

# use np.vectorize for efficient element-wise operations
combine = (
    lambda func, data, w: np.vectorize(func)(data) * w
)  # assumes that func only has data as arg (or as *args)


# order_type, symbol, quantity = order
def convert_to_order(num, ticker):
    if num > THRESHOLD_BUY:
        return [ticker, ORDER_SIZE, "BUY"]
    elif num < THRESHOLD_SELL:
        return [ticker, ORDER_SIZE, "SELL"]

    return [ticker, ORDER_SIZE, "HOLD"]


# Care for race-condition-like edge case. We do not control for way orders are arranged.

# Generate orders

for ticker, data in tqdm(zip(tickers, stock_data)):
    # element-wise linear combination
    res = np.add(combine(func, data, w) for func, w in weights.items())
    for idx, date_output in enumerate(res):
        orders[idx].append(convert_to_order(date_output))

# TO-DO: TIE INTO MARKET TESTER

print(f"Done. Took {time.time-start_time:.2f} seconds.")
# https://python-course.eu/numerical-programming/linear-combinations-in-python.php
