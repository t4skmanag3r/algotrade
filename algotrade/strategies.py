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
            self.buy_signals = (
                (self.short > self.long) & (self.ma < self.df.close) & (self.rsi < 65)
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


class Ichimoku(Strategy):
    def __init__(self, low_period=9, med_period=26, high_period=52) -> None:
        super().__init__()
        self.low_period = low_period
        self.med_period = med_period
        self.high_period = high_period
        self.legend = []

    def _calc(self, df):
        from ta.trend import IchimokuIndicator

        self.df = df
        self.ichimoku = IchimokuIndicator(
            high=df.high,
            low=df.low,
            window1=self.low_period,
            window2=self.med_period,
            window3=self.high_period,
        )
        cloud_price_buy, cloud_price_sell = self.cloud_price()
        cloud_color_buy, cloud_color_sell = self.cloud_color()
        cloud_dirrection_buy, cloud_dirrection_sell = self.cloud_dirrection()
        cloud_growth_buy, cloud_growth_sell = self.cloud_growth()
        conv_base_above_buy, conv_base_above_sell = self.conv_base_above()
        conv_base_cross_buy, conv_base_cross_sell = self.conv_base_cross()
        conv_dirrection_buy, conv_dirrection_sell = self.conv_dirrection()
        (
            conv_base_diff_growth_buy,
            conv_base_diff_growth_sell,
        ) = self.conv_base_diff_growth()
        lag_price_buy, lag_price_sell = self.lag_price()
        lag_cloud_buy, lag_cloud_sell = self.lag_cloud()

        self.buy_signals = (
            cloud_price_buy
            & cloud_color_buy
            & cloud_dirrection_buy
            & cloud_growth_buy
            & conv_base_above_buy
            # & conv_base_cross_buy
            & conv_dirrection_buy
            & conv_base_diff_growth_buy
            & lag_price_buy
            & lag_cloud_buy
        )
        self.sell_signals = cloud_price_sell | lag_price_sell

    def plot(self, days=None):
        from algotrade.ploting import Ichimoku

        df = self.df

        if days is not None:
            df = self.df[-days:]

        Ichimoku(df).plot()

    def cloud_price(self):
        leadspan_a = self.ichimoku.ichimoku_a().shift(self.med_period)
        leadspan_b = self.ichimoku.ichimoku_b().shift(self.med_period)

        cloud_high = np.max(pd.concat([leadspan_a, leadspan_b], axis=1).T)
        price_low = np.min(self.df[["open", "close"]], axis=1)
        cloud_low = np.min(pd.concat([leadspan_a, leadspan_b], axis=1).T)
        price_high = np.max(self.df[["open", "close"]], axis=1)

        buy_signals = price_low > cloud_high
        sell_signals = price_high <= cloud_low
        return buy_signals, sell_signals

    def cloud_color(self):
        leadspan_a = self.ichimoku.ichimoku_a()
        leadspan_b = self.ichimoku.ichimoku_b()

        buy_signals = leadspan_a > leadspan_b
        sell_signals = leadspan_a <= leadspan_b
        return buy_signals, sell_signals

    def cloud_dirrection(self):
        leadspan_a = self.ichimoku.ichimoku_a()
        leadspan_b = self.ichimoku.ichimoku_b()

        buy_signals = (leadspan_a - leadspan_a.shift(1)) + (
            leadspan_b - leadspan_b.shift(1)
        ) > 0
        sell_signals = (leadspan_a - leadspan_a.shift(1)) + (
            leadspan_b - leadspan_b.shift(1)
        ) <= 0
        return buy_signals, sell_signals

    def cloud_growth(self):
        leadspan_a = self.ichimoku.ichimoku_a()
        leadspan_b = self.ichimoku.ichimoku_b()

        buy_signals = abs(leadspan_a - leadspan_b) > abs(
            leadspan_a.shift(1) - leadspan_b.shift(1)
        )
        sell_signals = abs(leadspan_a - leadspan_b) < abs(
            leadspan_a.shift(1) - leadspan_b.shift(1)
        )
        return buy_signals, sell_signals

    def conv_base_above(self):
        conv_line = self.ichimoku.ichimoku_conversion_line()
        base_line = self.ichimoku.ichimoku_base_line()

        buy_signals = conv_line > base_line
        sell_signals = conv_line <= base_line
        return buy_signals, sell_signals

    def conv_base_cross(self):
        conv_line = self.ichimoku.ichimoku_conversion_line()
        base_line = self.ichimoku.ichimoku_base_line()

        buy_signals = (conv_line > base_line) & (
            conv_line.shift(1) < base_line.shift(1)
        )
        sell_signals = (conv_line < base_line) & (
            conv_line.shift(1) > base_line.shift(1)
        )
        return buy_signals, sell_signals

    def conv_dirrection(self):
        conv_line = self.ichimoku.ichimoku_conversion_line()

        buy_signals = (conv_line - conv_line.shift(1)) > 0
        sell_signals = (conv_line - conv_line.shift(1)) < 0
        return buy_signals, sell_signals

    def conv_base_diff_growth(self):
        conv_line = self.ichimoku.ichimoku_conversion_line()
        base_line = self.ichimoku.ichimoku_base_line()

        buy_signals = abs(conv_line - base_line) > abs(
            conv_line.shift(1) - base_line.shift(1)
        )
        sell_signals = abs(conv_line - base_line) < abs(
            conv_line.shift(1) - base_line.shift(1)
        )
        return buy_signals, sell_signals

    def lag_price(self):
        lag = self.df.close
        past_price = np.max(self.df[["open", "close"]].shift(self.med_period), axis=1)
        buy_signals = lag > past_price
        sell_signals = lag < past_price
        return buy_signals, sell_signals

    def lag_cloud(self):
        lag = self.df.close
        leadspan_a = self.ichimoku.ichimoku_a()
        leadspan_b = self.ichimoku.ichimoku_b()
        cloud_high = np.max(pd.concat([leadspan_a, leadspan_b], axis=1).T)
        cloud_low = np.min(pd.concat([leadspan_a, leadspan_b], axis=1).T)

        buy_signals = lag > cloud_high.shift(self.med_period)
        sell_signals = lag < cloud_low.shift(self.med_period)
        return buy_signals, sell_signals


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
