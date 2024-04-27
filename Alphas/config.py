import os
import json
import pandas as pd
import random

root = "Alphas"


class Config:
    def __init__(self, path="Data/Config", rt=True):
        self.config_root = os.path.join(root, path) if rt else path
        with open(os.path.join(self.config_root, "config.json")) as f:
            self.cfg = json.loads(f.read())
            self.weights = self.cfg["weights"]
            self.opt = self.cfg["options"]
            self.orders = None
            self.exog = self.cfg["exogenous_opt"]
            self.broker_opt = self.cfg["broker_opt"]
            self.fmt = self.exog["_fmt"]
            self.INITIAL_CAPITAL = self.exog["initial_capital"]

    def setOrders(self, orders: pd.DataFrame):
        self.orders = orders.dropna(inplace=False, axis=0)

    def setWeights(self, weights: dict, updateJSON: bool = False):
        self.weights = weights
        self.cfg["weights"] = weights
        if updateJSON:
            self.dumpJSON()

    def dumpJSON(self):
        with open(os.path.join(self.config_root, "config.json"), "w") as fp:
            json.dump(self.cfg, fp)


def random_thresholds(
    low_sell=1.1, high_sell=1.28, low_buy=1.28, high_buy=1.35, sigfigs=4
):
    #  "threshold_buy": 1.3,
    # "threshold_sell": 1.26,
    x = 10**sigfigs
    sell_threshold = float(random.randrange(low_sell * x, high_sell * x)) / x
    buy_threshold = (
        float(random.randrange(max(low_buy, sell_threshold) * x, high_buy * x)) / x
    )

    return (sell_threshold, buy_threshold)


cfg = Config()
