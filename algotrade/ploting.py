import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def plot(self):
        import matplotlib.ticker as plticker

        f2 = plt.figure(figsize=(24, 9))
        ax2 = plt.axes()

        loc = plticker.AutoLocator()
        ax2.xaxis.set_major_locator(loc)
        return f2, ax2


class Indicator:
    def __init__(self, df: pd.DataFrame, indicator, timeframe) -> None:
        self.df = df
        self.data = self._calc(indicator, timeframe)
        self.legend = []
        self.plot()

    def _calc(self, indicator, timeframe) -> np.ndarray:
        return indicator(self.df.close, timeframe)

    def plot(self) -> None:
        plt.plot(self.df.index.astype(str), self.data, color="c")


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
        plt.plot(self.df.index.astype(str), self.conv_line, color="darkorange")
        plt.plot(self.df.index.astype(str), self.base_line, color="dodgerblue")
        plt.plot(self.df.index.astype(str), self.leadspan_a.shift(26), color="green")
        plt.plot(self.df.index.astype(str), self.leadspan_b.shift(26), color="red")

        plt.plot(self.df.index.astype(str), self.df.close.shift(-26), color="plum")
        plt.fill_between(
            self.df.index.astype(str),
            self.leadspan_a.shift(26).values,
            self.leadspan_b.shift(26).values,
            alpha=0.5,
            color="lightblue",
        )
        plt.legend(self.legend)


class Macd(Plotting):
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

        super().__init__(df)
        self.data = self._calc(MACD, timeframe_short, timeframe_long, timeframe_signal)
        self.legend = []

    def _calc(self, indicator, timeframe_short, timeframe_long, timeframe_signal):
        macd = indicator(
            self.df.close, timeframe_long, timeframe_short, timeframe_signal
        )
        self.macd = macd.macd()
        self.macd_signal = macd.macd_signal()
        self.macd_diff = macd.macd_diff()

    def plot(self):
        f2, ax2 = super().plot()
        ax2.plot(self.macd.index.astype(str), self.macd.values, color="green")
        ax2.plot(
            self.macd_signal.index.astype(str), self.macd_signal.values, color="red"
        )
        ax2.bar(
            self.macd_diff.index.astype(str), self.macd_diff.values, color="lightblue"
        )
        ax2.legend(["macd", "macd_signal", "macd_diff"])
        return ax2


class RSI(Plotting):
    def __init__(self, df, rsi=None, rsi_2=None) -> None:
        super().__init__(df)
        if rsi is not None:
            self.rsi = rsi
        else:
            self._calc()
        self.rsi_2 = rsi_2

    def _calc(self, timeframe=14):
        from ta.momentum import rsi

        self.rsi = rsi(self.df["close"], window=timeframe)

    def plot(self):
        f2, ax2 = super().plot()

        ax2.plot(self.df.index.astype(str), self.rsi, color="cyan")
        ax2.legend(["rsi"])
        if self.rsi_2 is not None:
            ax2.plot(self.df.index.astype(str), self.rsi_2, color="orange")
            ax2.legend(["rsi", "rsi_2"])
        ax2.axhline(y=70)
        ax2.axhline(y=30)
        return ax2

class Stochastic(Plotting):
    def __init__(self, df, stoch_stoch=None, stoch_signal=None) -> None:
        super().__init__(df)
        if stoch_stoch is None or stoch_signal is None:
            self._calc()
        else:
            self.stoch_stoch= stoch_stoch
            self.stoch_signal = stoch_signal

    def _calc(self, timeframe=14, smooth=3):
        from algotrade.custom_indicators import SlowStochasticOscillator as StOc

        stoch = StOc(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=timeframe, smooth_window=smooth)
        self.stoch_stoch = stoch.stoch()
        self.stoch_signal = stoch.stoch_signal()

    def plot(self):
        f2, ax2 = super().plot()

        ax2.plot(self.df.index.astype(str), self.stoch_stoch, color="green")
        ax2.plot(self.df.index.astype(str), self.stoch_signal, color="red")

        ax2.legend(["stoch", "stoch_signal"])
        ax2.axhline(y=80)
        ax2.axhline(y=20)
        return ax2
