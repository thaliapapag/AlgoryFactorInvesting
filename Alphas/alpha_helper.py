import pandas as pd
import numpy as np
import os

# from helper import *
from ALPHA_TEST import *  # we should move the general helper functions into helper.py
from data_load import load_data, load_alpha_helper_data
from helper import vwap

# dict.update()
# I'm going to rewrite these later

root = "Data"
alpha_path = "alpha_main.json"
helper_path = "alpha_src.json"

# Columns: stocks. Index: time
stock_history = load_data()

alpha_dict = (
    load_data(alpha_path) if os.path.exists(os.path.join(root, alpha_path)) else {}
)
# Dictionary of DataFrames with each DataFrame representing a helper alpha (metric that must be ranked cross-sectionally or across industry). Columns: stocks. Index: time
alpha_helper_dict = (
    load_alpha_helper_data(helper_path)
    if os.path.exists(os.path.join(root, helper_path))
    else {}
)

"""
alpha_dict is a hashmap of DataFrames
Each DataFrame 

alpha_helper_dict is a hashmap (alpha) of a hashmap (sub-alphas/helper alphas) of DataFrames
"""


def alpha_77(hst, tickers: list):
    # we need to map this to everything
    helper_root = alpha_helper_dict["alpha_77"]
    alpha_root = alpha_dict["alpha_77"]

    for ticker in tickers:
        pass

    vwap = vwap(hst)
    # (high + low) / 2) + high
    hst["high_low"] = hst["High"] + hst["Low"]
    hst["high_low"].apply(lambda x: np.divide(x / 2))
    hst["decayed_avg"] = hst["decayed_avg"] + vwap
    # linear decay over period rounded to 20 days
    hst["a77_1"] = decay_linear(hst["decayed_avg"], 20)

    # we need to cross-sectional rank both linear decays
    helper_root("ticker")
    hst["rank_1"] = rank(hst["a77_1"])

    hst["adv40"] = adv(hst, 40)

    d = 3  # Round to 3 days
    time_series_correlation = hst["high_low"].rolling(window=3).corr(hst["adv40"])
    hst["a77_2"] = decay_linear(time_series_correlation, 6)  # round to 6 days

    return pd.DataFrame(hst[["a77_1", "a77_2"]])


def alpha_101(hst):
    hst["alpha_101"] = (hst["Close"] - hst["Open"]) / (
        (hst["High"] - hst["Low"]) + 10**-8
    )
    return hst["alpha_101"]
