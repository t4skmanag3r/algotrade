import pandas as pd
import numpy as np
from abc import ABC


def _combineStrategies(
    df,
    buy_strategies,
    buy_weights,
    sell_strategies,
    sell_weights,
    buy_threshold,
    sell_threshold,
):  # deprecated
    """
    Combines given strategies into one, based on given weights and thresholds,
    works by adding/subtrating from total score and applying threshold from trigering Buy/Sell signals,
    Disclaimer (Values of weights and thresholds must be possitive as they are flipped in operation)

    Args:
        buy_strategies : list(str)
            must be a list of column names calculated by applying strategies - list(True/Flase), used for buy signals
        buy_weights : list(int)
            must be a list of weights tied to buy_strategies used for adding to the total score
        sell_strategies : list(str)
            must be a list of column names calculated by applying strategies - list(True/Flase), used for sell signals
        sell_weights : list(int)
            must be a list of weights tied to buy_strategies used for subtracting from the total score  (non negative)
        buy_threshold:
            must be a threshold needed to pass to generate a buy signal
        sell_threshold:
            must be a threshold needed to pass to generate a sell signal (non negative)

    Returns:
        buy_signals : list(bool)
            list of buy signals True/False, based on score and threshold
        sell_signals : list(bool)
            list of sell signals True/False based score and threshold
    """

    def add_sub_score(
        row,
    ):  # Adds or subtracts from total score by set weights depending on True/False values
        score = 0
        for buy_signal, buy_weight in zip(buy_strategies, buy_weights):
            if row[buy_signal]:
                score += buy_weight
        for sell_signal, sell_weight in zip(sell_strategies, sell_weights):
            if row[sell_signal]:
                score -= sell_weight
        return score

    scores = list(df.apply(add_sub_score, axis=1).values)
    # Applying threshold and returning True only when total score above or below this threshold
    buy_signals = [True if score > buy_threshold else False for score in scores]
    sell_signals = [True if score < -sell_threshold else False for score in scores]
    return buy_signals, sell_signals


class Strategy(ABC):
    def __init__(self) -> None:
        pass

    def _calc(self, df):
        self.buy_signals = []
        self.sell_signals = []

    def apply(self, df):
        self._calc(df)
        return self.buy_signals, self.sell_signals


class StrategySimple(Strategy):
    def __init__(self, indicator, periods_short, periods_long, name):
        self.periods_short = periods_short
        self.periods_long = periods_long
        self.name = name
        self.indicator = indicator
        self.legend = [f"{name+str(periods_short)}", f"{name+str(periods_long)}"]

    def _calc(self, df):
        self.df = df
        self.short = self.indicator(df.close, window=self.periods_short)
        self.long = self.indicator(df.close, window=self.periods_long)
        self.buy_signals = self.short > self.long
        self.sell_signals = self.short < self.long

    def plot(self, days=None):
        import matplotlib.pyplot as plt

        if days is None:
            plt.plot(self.df.index, self.short)
            plt.plot(self.df.index, self.long)
        else:
            plt.plot(self.df.index[-days:], self.short[-days:])
            plt.plot(self.df.index[-days:], self.long[-days:])


class WeightedMovingAverage(StrategySimple):
    def __init__(self, periods_short=25, periods_long=32, name="wma"):
        from ta.trend import wma_indicator

        super().__init__(wma_indicator, periods_short, periods_long, name)


class ExponentialMovingAverage(StrategySimple):
    def __init__(self, periods_short=25, periods_long=32, name="ema"):
        from ta.trend import ema_indicator

        super().__init__(ema_indicator, periods_short, periods_long, name)


class SMA200(StrategySimple):
    def __init__(self, indicator, periods_short, periods_long, name):
        super().__init__(indicator, periods_short, periods_long, name)
        self.legend = self.legend + ["ma200"]

    def _calc(self, df):
        super()._calc(df)
        from ta.trend import sma_indicator

        self.short = self.indicator(self.df.close, window=self.periods_short)
        self.long = self.indicator(self.df.close, window=self.periods_long)
        self.ma200 = sma_indicator(self.df.close, window=200)
        self.short.shift()

        self.buy_signals = (self.short > self.long) & (self.ma200 < self.df.close)
        self.sell_signals = self.short < self.long

    def plot(self, days=None):
        import matplotlib.pyplot as plt

        super().plot(days)
        if days is None:
            plt.plot(self.df.index, self.ma200)
        else:
            plt.plot(self.df.index[-days:], self.ma200[-days:])


class RSI(Strategy):
    def __init__(self, rsi_upper_thresh=70, rsi_lower_thresh=30) -> None:
        self.rsi_upper_thresh = rsi_upper_thresh
        self.rsi_lower_thresh = rsi_lower_thresh
        self.legend = []

    def _calc(self, df):
        from ta.momentum import rsi

        self.df = df

        self.rsi = rsi(df["close"], window=14)
        self.buy_signals = self.rsi < self.rsi_lower_thresh
        self.sell_signals = self.rsi > self.rsi_upper_thresh

    def plot(self, days=None):
        from algotrade.ploting import RSI

        if days is None:
            RSI(df=self.df, rsi=self.rsi).plot()
        else:
            RSI(df=self.df[-days:], rsi=self.rsi[-days:]).plot()


class RSI_MACD(Strategy):
    def __init__(
        self, rsi_upper_thresh=70, rsi_lower_thresh=30, macd_short=12, macd_long=26
    ) -> None:
        self.rsi_upper_thresh = rsi_upper_thresh
        self.rsi_lower_thresh = rsi_lower_thresh
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.legend = []

    def _calc(self, df):
        from ta.momentum import rsi
        from ta.trend import MACD

        self.df = df

        self.rsi = rsi(df["close"], window=14)
        self.macd = MACD(
            df["close"], window_fast=self.macd_short, window_slow=self.macd_long
        )
        self.macd_buy_signals = self.macd.macd_signal() > self.macd.macd()
        self.macd_sell_signals = self.macd.macd_signal() < self.macd.macd()
        self.rsi_buy_signals = self.rsi < self.rsi_lower_thresh
        self.rsi_sell_signals = self.rsi > self.rsi_upper_thresh
        self.buy_signals = []
        self.sell_signals = []
        rsi_triger_buy = False
        rsi_triger_sell = False
        bought = [0]
        for i, (
            signal_rsi_buy,
            signal_macd_buy,
            signal_rsi_sell,
            signal_macd_sell,
        ) in enumerate(
            zip(
                self.rsi_buy_signals,
                self.macd_buy_signals,
                self.rsi_sell_signals,
                self.macd_sell_signals,
            )
        ):
            if signal_rsi_buy:
                rsi_triger_buy = True
            if rsi_triger_buy and signal_macd_buy:
                self.buy_signals.append(True)
                bought.append(df.iloc[i].close)
                rsi_triger_buy = False
            else:
                self.buy_signals.append(False)
            if signal_rsi_sell:
                rsi_triger_sell = True
            if (
                rsi_triger_sell
                and signal_macd_sell
                or self.df.iloc[i].close < bought[-1]
            ):
                self.sell_signals.append(True)
                rsi_triger_sell = False
            else:
                self.sell_signals.append(False)

    def plot(self, days=None):
        from algotrade.ploting import Macd, RSI

        df = self.df
        rsi = self.rsi
        if days:
            df = df[-days:]
            rsi = rsi[-days:]
        Macd(
            df=df, timeframe_short=self.macd_short, timeframe_long=self.macd_long
        ).plot()
        RSI(df=df, rsi=rsi).plot()


class MACD(SMA200):  # Apply Moving Average and MACD strategy
    def __init__(
        self, periods_short=12, periods_long=26, sma=False, sma_period=100, name="macd"
    ):
        self.periods_short = periods_short
        self.periods_long = periods_long
        self.sma = sma
        self.sma_period = sma_period
        self.legend = ["ma" + str(sma_period)]

    def _calc(self, df):
        from ta.trend import macd_signal, macd, sma_indicator
        from ta.momentum import rsi

        self.df = df
        self.short = macd(
            self.df.close, window_slow=self.periods_short, window_fast=self.periods_long
        )
        self.long = macd_signal(
            self.df.close, window_slow=self.periods_short, window_fast=self.periods_long
        )
        self.ma = sma_indicator(self.df.close, window=self.sma_period)
        self.rsi = rsi(self.df.close)
        self.short.shift()

        if self.sma:
            self.buy_signals = (
                (self.short > self.long) & (self.ma < self.df.close) & (self.rsi < 65)
            )
        else:
            self.buy_signals = self.short > self.long & (self.ma < self.df.close) & (
                self.rsi < 65
            )
        self.sell_signals = self.short < self.long

    def plot(self, days=None):
        from algotrade.ploting import Macd, Sma, RSI

        df = self.df
        if days is not None:
            df = self.df[-days:]

        Sma(df, timeframe=self.sma_period).plot()
        Macd(df).plot()
        RSI(df).plot()


class ChandelierExitRSI(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def _calc(self):

        self.buy_signals = []
        self.sell_signals = []


class Ichimoku:
    def __buy_sell_filter(
        df, var_pos, var_neg
    ):  # filters buying selling if var1 > var2  and vice versa
        buy_signals = df[var_pos] > df[var_neg]
        sell_signals = df[var_pos] < df[var_neg]
        return buy_signals.values, sell_signals.values

    @classmethod
    def strategy_Ichimoku_conv_base(cls, df):
        return cls.__buy_sell_filter(df, "convline", "baseline")

    def strategy_Ichimoku_cloud_price(df):
        buy_signals = (
            ((df["close"] > df["leadspan_a"]) == True)
            & ((df["leadspan_a"] > df["leadspan_b"]) == True)
        ) | (
            ((df["close"] > df["leadspan_b"]) == True)
            & ((df["leadspan_a"] < df["leadspan_b"]) == True)
        )
        sell_signals = (
            ((df["close"] < df["leadspan_a"]) == True)
            & ((df["leadspan_a"] > df["leadspan_b"]) == True)
        ) | (
            ((df["close"] < df["leadspan_b"]) == True)
            & ((df["leadspan_a"] < df["leadspan_b"]) == True)
        )
        return buy_signals.values, sell_signals.values

    @classmethod
    def strategy_Ichimoku_leadspan(cls, df):
        return cls.__buy_sell_filter(df, "leadspan_a", "leadspan_b")


def applyStrategiesIchimoku(df):
    df["buy_conv"], df["sell_conv"] = Ichimoku.strategy_Ichimoku_conv_base(df)
    df["buy_cloud"], df["sell_cloud"] = Ichimoku.strategy_Ichimoku_cloud_price(df)
    df["buy_leadspan"], df["sell_leadspan"] = Ichimoku.strategy_Ichimoku_leadspan(df)
    return df


# def main():
#     import general
#     import calculations

#     df = general.getData('TSLA', '2016-01-01')
#     df = calculations.calculateData(df)

# buy_strategies = ["buy_conv", "buy_cloud", "buy_leadspan"]
# buy_weights = [4, 2, 4]
# sell_strategies = ["sell_conv", "sell_cloud", "sell_leadspan"]
# sell_weights = [1, 1, 2]
# buy_threshold = 4
# sell_threshold = 0
# comb_buy, comb_sell = combineStrategies(df, buy_strategies, buy_weights, sell_strategies, sell_weights, buy_threshold, sell_threshold)
# print(comb_buy, comb_sell)

# strategies = ["convbase_div", "leadspan_div", "cloudprice_div"]
# weights = [1, 1, 1]
# biases = [0, 0, 0]
# buy_threshold = 0.05
# sell_threshold = 0.05

# buy_signals, sell_signals = combineStrategies(df, strategies, weights, biases, buy_threshold, sell_threshold)
# buy_dates, sell_dates = general.getBuySellDates(df, buy_signals, sell_signals)
# print(buy_dates, sell_dates)

# buy_signals, sell_signals = strategy_MA_MACD(df)
# buy_dates, sell_dates = general.getBuySellDates(df, buy_signals, sell_signals)
# print(buy_dates, sell_dates)
# print(strategyList())

# if __name__ == '__main__':
#     main()
