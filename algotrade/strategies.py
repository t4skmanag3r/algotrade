import pandas as pd
import numpy as np


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


class Strategy:
    def __init__(self) -> None:
        pass

    def _calc(self, df):
        self.df = df
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
        super()._calc(df)
        self.short = self.indicator(self.df.close, window=self.periods_short)
        self.long = self.indicator(self.df.close, window=self.periods_long)
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
    def __init__(self, periods_short=9, periods_long=18, name="wma"):
        from ta.trend import wma_indicator

        super().__init__(wma_indicator, periods_short, periods_long, name)


class ExponentialMovingAverage(StrategySimple):
    def __init__(self, periods_short=9, periods_long=18, name="ema"):
        from ta.trend import ema_indicator

        super().__init__(ema_indicator, periods_short, periods_long, name)


class MovingAverageAnd200SMA(StrategySimple):
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
        self.buy_signals = (self.short > self.long) & (
            (self.short.shift(1) < self.long.shift(1)) & (self.ma200 < self.df.close)
        )
        self.sell_signals = self.short < self.long

    def plot(self, days=None):
        import matplotlib.pyplot as plt

        super().plot(days)
        if days is None:
            plt.plot(self.df.index, self.ma200)
        else:
            plt.plot(self.df.index[-days:], self.ma200[-days:])


def strategy_MA_MACD(df):  # Apply Moving Average and MACD strategy
    df = df.copy()
    from algotrade.calculations import calcSMA, calcMACD

    df = calcSMA(df, 200)
    df = calcMACD(df)
    buy_signals = ((df["close"] > df["MA200"]) == True) & (
        (df["MACD"] > df["MACDSignal"]) == True
    )
    sell_signals = (df["MACD"] < df["MACDSignal"]) == True
    return buy_signals.values, sell_signals.values


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
