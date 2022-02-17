import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Indicator:
    def __init__(self, df: pd.DataFrame, indicator, timeframe) -> None:
        self.df = df
        self.data = self._calc(indicator, timeframe)
        self.legend = []
        self.plot()

    def _calc(self, indicator, timeframe) -> np.ndarray:
        return indicator(self.df.close, timeframe)

    def plot(self) -> None:
        print(self.data)
        plt.plot(self.data)


class Sma(Indicator):
    def __init__(self, df: pd.DataFrame, timeframe: int) -> None:
        from ta.trend import sma_indicator

        super().__init__(df, sma_indicator, timeframe)
        self.legend = ["sma_" + str(timeframe)]


class Ema(Indicator):
    def __init__(self, df: pd.DataFrame, timeframe: int) -> None:
        from ta.trend import ema_indicator

        super().__init__(df, ema_indicator, timeframe)
        self.legend = ["ema" + str(timeframe)]


class Ichimoku:
    def __init__(self, df: pd.DataFrame) -> None:
        from ta.trend import IchimokuIndicator

        self.df = df
        self.data = self._calc(IchimokuIndicator)
        self.legend = ["conv_line", "base_line", "leadspan_a", "leadspan_b"]
        self.plot()

    def _calc(self, indicator) -> None:
        ichimoku = indicator(
            self.df[["open", "high", "low", "close"]].max(axis=1),
            self.df[["open", "high", "low", "close"]].min(axis=1),
        )
        self.conv_line = ichimoku.ichimoku_conversion_line()
        self.base_line = ichimoku.ichimoku_base_line()
        self.leadspan_a = ichimoku.ichimoku_a()
        self.leadspan_b = ichimoku.ichimoku_b()

    def plot(self) -> None:
        plt.plot(self.conv_line, color="green")
        plt.plot(self.base_line, color="red")
        plt.plot(self.leadspan_a, color="lightblue")
        plt.plot(self.leadspan_b, color="tan")
        plt.fill_between(
            self.leadspan_a.index,
            self.leadspan_a.values,
            self.leadspan_b.values,
            alpha=0.5,
            color="lightblue",
        )


class Macd:
    """
    Plots MACD graph

    Args:
        timeframe_short : int
            short timeframe used to calculate macd
        timeframe_long : int
            long timeframe used to calculate macd
        timeframe_signal : int
            signal timeframe used to calculate macd signal
    """

    def __init__(
        self,
        df: pd.DataFrame,
        timeframe_short=12,
        timeframe_long=26,
        timeframe_signal=9,
    ) -> None:
        from ta.trend import MACD

        self.df = df
        self.data = self._calc(MACD, timeframe_short, timeframe_long, timeframe_signal)
        self.legend = []
        self.plot()

    def _calc(self, indicator, timeframe_short, timeframe_long, timeframe_signal):
        macd = indicator(
            self.df.close, timeframe_short, timeframe_long, timeframe_signal
        )
        self.macd = macd.macd()
        self.macd_signal = macd.macd_signal()
        self.macd_diff = macd.macd_diff()

    def plot(self):
        f2 = plt.figure(figsize=(24, 9))
        ax2 = plt.axes()
        ax2.plot(self.macd, color="green")
        ax2.plot(self.macd_signal, color="red")
        ax2.bar(self.macd_diff.index, self.macd_diff.values, color="lightblue")
        f2.legend(["macd", "macd_signal", "macd_diff"])
        return ax2
