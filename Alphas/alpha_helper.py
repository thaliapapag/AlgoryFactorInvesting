import pandas as pd
import numpy as np
from helper import *

# dict.update()
# I'm going to rewrite these later


def alpha_77(hst):
    vwap = vwap(hst)
    # (high + low) / 2) + high
    hst["high_low"] = hst["High"] + hst["Low"]
    hst["high_low"].apply(lambda x: np.divide(x / 2))
    hst["decayed_avg"] = hst["decayed_avg"] + vwap
    # linear decay over period rounded to 20 days
    hst["a77_1"] = decay_linear(hst["decayed_avg"], 20)

    # we need to cross-sectional rank both linear decays

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
