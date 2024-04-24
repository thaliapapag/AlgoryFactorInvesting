import os
import json
import pandas as pd

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

    def setWeights(self, weights: dict):
        self.weights = weights
        self.cfg["weights"] = weights

    def dumpJSON(self):
        with open("result.json", "w") as fp:
            json.dump(self.cfg, fp)


cfg = Config()
