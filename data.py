import pandas as pd
import torch
import numpy as np
import csv
from decimal import ROUND_HALF_UP, Decimal


class Stock(object):
    def __init__(self, code):
        self.code = int(code)
        self.prices = []
        self.vols = []
        self.ranks = []
        self.targets = []
        return

    def add(self, price, vol, target, rank):
        self.prices = self.prices[-100:] + [float(price)]
        self.vols = self.vols[-100:] + [float(vol)]
        self.targets = self.targets[-100:] + [float(target)]
        self.ranks = self.ranks[-100:] + [float(rank)]

    def features(self):
        "get a bunch of features, starting with adjusted prices"
        self.prices = np.nan_to_num(self.prices)
        sma_30 = np.mean(self.prices[-30:])
        sma_60 = np.mean(self.prices[-60:])
        sma_90 = np.mean(self.prices[-90:])
        close = self.prices[-1]
        volm = self.vols[-1]
        rvalue = (self.prices[-1] - self.prices[-2]) / \
            self.prices[-2] if len(self.prices) > 2 else 0
        pct_10 = (self.prices[-1] - self.prices[-10]) / \
            self.prices[-10] if len(self.prices) > 10 else 0
        pct_21 = (self.prices[-1] - self.prices[-21]) / \
            self.prices[-21] if len(self.prices) > 21 else 0
        pct_63 = (self.prices[-1] - self.prices[-63]) / \
            self.prices[-63] if len(self.prices) > 63 else 0
        return [close, volm, close/sma_30, rvalue, sma_30, sma_60, sma_90, pct_10, pct_21, pct_63]


class TSDataset(object):
    """ Date keyed dataset for online training"""

    def __init__(self, name, get_next=None):
        self.name = name
        self.idx = 0
        self.cur_time = None
        if get_next is not None:
            self.get_next = get_next
        return

    def get_next(self):
        """
        To be provided by the client. this method should return timestamp, [stock]
        """
        raise NotImplemented("Initialization requires next method")

    def __iter__(self):
        for (ts, rows) in self.get_next():
            if not ts:
                continue
            if self.cur_time is None:
                self.cur_time = ts
            else:
                assert ts > self.cur_time, f"Time must be in right order: {ts}, {self.cur_time}"
            self.cur_time = ts
            self.idx += 1
            yield ts, rows

    def __len__(self):
        return self.idx


def gen_stocks():
    df = pd.read_csv("./data/stock_list.csv")
    syms = sorted(df.SecuritiesCode.values)
    lookups = {sym: i for i, sym in enumerate(syms)}
    stocks = [Stock(c) for c in syms]
    return syms, lookups, stocks


REVERSE_LOOKUP, LOOKUP, STOCKS = gen_stocks()


def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[:, "Date"] = pd.to_datetime(
        price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(
        generate_adjusted_close).reset_index(drop=True)

    return price


def from_train():
    """This function must provide per timestamp (day) the full data"""
    df = pd.read_csv("./data/train_files/stock_prices.csv", index_col="RowId")
    df = adjust_price(df).fillna(0)
    print(df.head)
    df['Date'] = pd.to_datetime(df.Date)
    for ts, ds in df.groupby('Date'):
        out = []
        for rank, (idx, row) in enumerate(ds.sort_values('Target', ascending=False).iterrows()):
            stock = STOCKS[LOOKUP[row.SecuritiesCode]]
            stock.add(row.AdjustedClose, row.Volume, row.Target, rank)
            out.append(stock)
        yield ts, out


DSTrain = TSDataset("DSTrain", from_train)

if __name__ == "__main__":
    cnt = 0
    for ts, rows in DSTrain:
        print(ts)
        print(len(rows))
        print(rows[10].features())
        cnt += 1
        if cnt > 100:
            break
