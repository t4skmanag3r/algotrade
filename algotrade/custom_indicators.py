from ta.momentum import StochasticOscillator as StOc
import pandas as pd

class SlowStochasticOscillator(StOc):
    """Slow Stochastic Oscillator (MODIFIED version of StochasticOscilator from TA library)

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

    Args:
        close(pandas.Series): dataset 'Close' column.
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window(int): n period.
        smooth_window(int): sma period over stoch_k.
        fillna(bool): if True, fill nan values.
    """
    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        smooth_window: int = 3,
        fillna: bool = False,
    ):
        super().__init__(high=high, low=low, close=close, window=window, smooth_window=smooth_window, fillna=fillna)

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        smin = self._low.rolling(self._window, min_periods=min_periods).min()
        smax = self._high.rolling(self._window, min_periods=min_periods).max()
        self._stoch_k = (100 * (self._close - smin) / (smax - smin)).rolling(self._smooth_window).mean()