import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# from helper import *
from ALPHA_TEST import *  # we should move the general helper functions into helper.py
from data_load import load_stock_history_data, load_alpha_helper_data, load_tickers
from helper import vwap, adv, decay_linear

# dict.update()
# I'm going to rewrite these later

cwd_folder = os.getcwd().split("\\")[-1]
root = "Alphas/Data" if cwd_folder == "AlgoryFactorInvesting" else "Data"

alpha_path = "alpha_main.json"
helper_path = "alpha_src.json"
alpha_helper_dict = {}
alpha_dict = {}
tickers = load_tickers(root)

# Columns: stocks. Index: time
stock_history = load_stock_history_data(root)

# alpha_dict = (
#     load_data(alpha_path) if os.path.exists(os.path.join(root, alpha_path)) else {}
# )
# # Dictionary of DataFrames with each DataFrame representing a helper alpha (metric that must be ranked cross-sectionally or across industry). Columns: stocks. Index: time
# alpha_helper_dict = (
#     load_alpha_helper_data(helper_path)
#     if os.path.exists(os.path.join(root, helper_path))
#     else {}
# )

"""
alpha_dict is a hashmap of DataFrames
Each DataFrame 

alpha_helper_dict is a hashmap (alpha) of a hashmap (sub-alphas/helper alphas) of DataFrames
"""


def alpha_77(tickers: list = tickers):
    #  min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
    #  rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))

    rankdf1 = pd.DataFrame()
    rankdf2 = pd.DataFrame()

    rankdf2.name = "a2"

    # create helper alpha #1
    # rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)
    for ticker in tqdm(tickers):
        # need to properly reindex. Instead of storing these as series, throw them into a outer merge dataframe with the time index
        hst = stock_history[ticker]
        # print(f"data shape: {hst.shape}")
        hst.name = ticker
        vwap_hst = vwap(hst)
        # (high + low) / 2) + high
        high_low = hst["High"] + hst["Low"]
        high_low.apply(lambda x: np.divide(x, 2))
        # linear decay over period rounded to 20 days
        a77_1 = decay_linear(vwap_hst, 20)
        rankdf1[ticker] = a77_1

        # print(rankdf1.index, vwap_hst.index, high_low.index, a77_1.index)

    # create helper alpha #2
    # rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    for ticker in tqdm(tickers):
        hst = stock_history[ticker]

        adv40 = adv(hst, 40)

        # Round to 3 days
        time_series_correlation = (
            high_low.rolling(window=3, center=True)
            .corr(adv40.dropna(axis=0), pairwise=True)
            .dropna(axis=0)
        )
        rankdf2[ticker] = decay_linear(time_series_correlation, 6)  # round to 6 days

    print(f"df1 {rankdf1}")
    print(f"df2 {rankdf2}, {rankdf2.index}, {rankdf2.values}")
    print(rankdf1.index.dtype, rankdf2.index.dtype)

    alpha = (
        pd.merge(rankdf1, rankdf2, left_index=True, right_index=True, how="outer")
        .dropna(axis=0)
        .min(axis=1)
    )

    print(alpha)

    return alpha


def alpha_101(hst):
    hst["alpha_101"] = (hst["Close"] - hst["Open"]) / (
        (hst["High"] - hst["Low"]) + 10**-8
    )
    return hst["alpha_101"]


if __name__ == "__main__":
    # test alpha
    alpha_77()
