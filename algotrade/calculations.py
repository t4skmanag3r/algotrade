import pandas as pd
import numpy as np


class HeikinAshi:
    def __init__(self, df) -> None:
        self.df = df
        self.calculate()

    def calculate(self):
        self.val_close = (
            self.df["open"] + self.df["high"] + self.df["low"] + self.df["close"]
        ) / 4
        val_open = []
        first = True
        for i, (_, row) in enumerate(self.df.iterrows()):
            if first == True:
                open = (row.open + row.close) / 2
                val_open.append(open)
                first = False
            else:
                open = (open + self.val_close[i - 1]) / 2
                val_open.append(open)
        # self.val_open = (self.df["open"].shift(1) + self.df["close"].shift(1)) / 2
        self.val_open = val_open
        temp_df = self.df.copy()
        temp_df["open"] = self.val_open
        temp_df["close"] = self.val_close
        self.val_high = temp_df[["open", "high", "close"]].max(axis=1)
        self.val_low = temp_df[["open", "low", "close"]].min(axis=1)

    def close(self):
        return self.val_close

    def open(self):
        return self.val_open

    def high(self):
        return self.val_high

    def low(self):
        return self.val_low


class ChandelierExit:
    def __init__(
        self,
        df,
        atr_period=1,
        atr_multiplier=1.85,
        use_close=False,
    ) -> None:
        self.df = df
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_close = False
        self.calculate()

    def calculate(self) -> None:
        from ta.volatility import AverageTrueRange

        atr = AverageTrueRange(
            self.df.high,
            self.df.low,
            self.df.close,
            window=self.atr_period,
            fillna=True,
        ).average_true_range()

        atr = atr * self.atr_multiplier

        highest_high = self.df.high.rolling(window=self.atr_period).max().values
        lowest_low = self.df.low.rolling(window=self.atr_period).min().values

        long_stop = highest_high - atr
        self.long_stop_prev = long_stop.shift(1)

        short_stop = lowest_low + atr
        self.short_stop_prev = short_stop.shift(1)

        cond_long = self.df.close.shift(1) > long_stop.shift(1)
        cond_long_true = np.vstack((long_stop, self.long_stop_prev)).T.max(1)
        self.long_stop = np.where(cond_long == True, cond_long_true, long_stop)

        cond_short = self.df.close.shift(1) < short_stop.shift(1)
        cond_short_true = np.vstack((short_stop, self.short_stop_prev)).T.min(1)
        self.short_stop = np.where(cond_short == True, cond_short_true, short_stop)

    def exit_long(self):
        return self.long_stop

    def exit_short(self):
        return self.short_stop

    def exit_long_prev(self):
        return self.long_stop_prev

    def exit_short_prev(self):
        return self.short_stop_prev


# -- Everything below this can be deprecated
def calculateData(df):
    """
    Calculates all technical indicators

    Args:
        df : pandas.DataFrame()
            dataframe of historical ticker data
    Returns:
        pandas.DataFrame()
            dataframe of calculated TA data + original data
    """
    from algotrade.strategies import applyStrategiesIchimoku, strategy_MA_MACD

    df = calcIchimoku(df)
    df = applyStrategiesIchimoku(df)
    df["Buy_MA_MACD"], df["Sell_MA_MACD"] = strategy_MA_MACD(df)

    df = calcSMA(df, 200)
    df = calcSMA(df, 100)
    df = calcSMA(df, 50)
    df = calcMACD(df)
    df = calcRSI(df)

    return df


def calcIchimoku(df):
    """
    Calculates all Ichimoku cloud indicators
    Read about Ichimoku: https://www.investopedia.com/terms/i/ichimoku-cloud.asp

    Args:
        df : pandas.DataFrame()
            dataframe of historical ticker data
    Returns:
        pandas.DataFrame()
            dataframe of calculated Ichimoku indicators + original data
    """
    df["pl9"], df["ph9"] = Ichimoku.calcPeriodMinMax(df, 9)
    df["pl26"], df["ph26"] = Ichimoku.calcPeriodMinMax(df, 26)
    df["pl52"], df["ph52"] = Ichimoku.calcPeriodMinMax(df, 52)

    df["convline"] = Ichimoku.calcConversionLine(df)
    df["baseline"] = Ichimoku.calcBaseLine(df)
    df["convgain"] = Ichimoku.calcConvGain(df)
    df["leadspan_a"] = Ichimoku.calcLeadingSpanA(df)
    df["leadspan_b"] = Ichimoku.calcLeadingSpanB(df)
    df["leadspan_div"] = Ichimoku.calcLeadSpanDivergence(df)
    df["convbase_div"] = Ichimoku.calcConvBaseDivergence(df)
    df["cloudprice_div"] = Ichimoku.calcCloudPriceDivergence(df)
    return df


def calcRSI(df):
    """
    Calculates RSI indicator
    Read about RSI: https://www.investopedia.com/terms/r/rsi.asp

    Args:
        df : pandas.DataFrame()
            dataframe of historical ticker data
    Returns:
        pandas.DataFrame()
            dataframe of calculated RSI indicators + original data
    """
    from warnings import filterwarnings

    filterwarnings("ignore")

    df["price_change"] = df["adjclose"].pct_change()
    df["Upmove"] = df["price_change"].apply(lambda x: x if x > 0 else 0)
    df["Downmove"] = df["price_change"].apply(lambda x: abs(x) if x < 0 else 0)
    df["avg_Up"] = df["Upmove"].ewm(span=19).mean()
    df["avg_Down"] = df["Downmove"].ewm(span=19).mean()
    df = df.dropna()
    df["RS"] = df["avg_Up"] / df["avg_Down"]
    df["RSI"] = df["RS"].apply(lambda x: 100 - (100 / (x + 1)))
    return df


def calcSMA(df, period_len):
    """
    Calculates Simple Moving Average (SMA)
    Read about SMA: https://www.investopedia.com/terms/s/sma.asp

    Args:
        df : pandas.DataFrame()
            dataframe of historical ticker data
        period_len : int
            length of moving average periods
    Returns:
        pandas.DataFrame()
            dataframe of calculated SMA of period_len + original data
    """
    df[f"MA{period_len}"] = df["close"].rolling(window=period_len).mean()
    return df


def calcEMA(df, period_len, col="close"):
    """
    Calculates Exponential Moving Average (EMA)
    Read about EMA: https://www.investopedia.com/terms/e/ema.asp

    Args:
        df : pandas.DataFrame()
            dataframe of historical ticker data
        period_len : int
            length of moving average periods
        col: str
            which col to use for calculation
    Returns:
        pandas.DataFrame()
            dataframe of calculated EMA of period_len + original data
    """
    prev_ema = None

    if col == "MACD":
        ma = df[col].head(26 + period_len).mean()

        def __calc(row):
            nonlocal prev_ema
            if period_len + 26 >= row.name + 1:
                return None
            elif prev_ema != None:
                ema_today = (row[col] * ((2 / (period_len + 1)))) + (
                    prev_ema * (1 - (2 / (period_len + 1)))
                )
                prev_ema = ema_today
                return ema_today
            else:
                ema_today = (row[col] * ((2 / (period_len + 1)))) + (
                    ma * (1 - (2 / (period_len + 1)))
                )
                prev_ema = ema_today
                return ema_today

    else:
        ma = df[col].head(period_len).mean()

        def __calc(row):
            nonlocal prev_ema
            if period_len >= row.name + 1:
                return None
            elif prev_ema != None:
                ema_today = (row[col] * ((2 / (period_len + 1)))) + (
                    prev_ema * (1 - (2 / (period_len + 1)))
                )
                prev_ema = ema_today
                return ema_today
            else:
                ema_today = (row[col] * ((2 / (period_len + 1)))) + (
                    ma * (1 - (2 / (period_len + 1)))
                )
                prev_ema = ema_today
                return ema_today

    copy_df = df.copy().reset_index()
    arr = copy_df.apply(__calc, axis=1).values
    return arr


def calcMACD(df):
    """
    Calculates MACD indicator
    Read about MACD: https://www.investopedia.com/terms/m/macd.asp

    Args:
        df : pandas.DataFrame()
            dataframe of historical ticker data
    Returns:
        Dataframe of calculated MACD indicators + original data
    """

    df["EMA26"] = calcEMA(df, 26)  # Exponential mooving average
    df["EMA12"] = calcEMA(df, 12)

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACDSignal"] = calcEMA(df, 9, col="MACD")
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]
    return df


class Ichimoku:
    @staticmethod
    def calcPeriodMinMax(
        df, period_len
    ):  # Gets the min and max of period closing price
        arr_min = np.array(
            df[["open", "high", "low", "close"]]
            .rolling(period_len)
            .min()
            .rolling(4, axis=1)
            .min()
            .close
        )
        arr_max = np.array(
            df[["open", "high", "low", "close"]]
            .rolling(period_len)
            .max()
            .rolling(4, axis=1)
            .max()
            .close
        )
        return arr_min, arr_max

    @staticmethod
    def calcConversionLine(df):
        arr = df.apply(lambda row: (row.ph9 + row.pl9) / 2, axis=1).values
        return arr

    @staticmethod
    def calcBaseLine(df):
        arr = df.apply(lambda row: (row.ph26 + row.pl26) / 2, axis=1).values
        return arr

    @staticmethod
    def calcConvGain(df):
        arr = df.apply(
            lambda row: (row.convline - row.baseline) / row.baseline, axis=1
        ).values
        return arr

    @staticmethod
    def calcLeadingSpanA(df):
        periods = 26
        arr = list(
            df.apply(lambda row: (row.convline + row.baseline) / 2, axis=1).values
        )
        arr = [None for _ in range(periods)] + arr
        arr = arr[:-periods]
        return np.array(arr, dtype=float)

    @staticmethod
    def calcLeadingSpanB(df):
        periods = 26
        arr = list(df.apply(lambda row: (row.ph52 + row.pl52) / 2, axis=1).values)
        arr = [None for _ in range(periods)] + arr
        arr = arr[:-periods]
        return np.array(arr, dtype=float)

    @staticmethod
    def calcConvBaseDivergence(df):
        arr = df.apply(
            lambda row: (row.convline - row.baseline) / row.baseline, axis=1
        ).values
        return arr

    @staticmethod
    def calcLeadSpanDivergence(df):
        arr = df.apply(
            lambda row: (row.leadspan_a - row.leadspan_b) / row.leadspan_b, axis=1
        ).values
        return arr

    @staticmethod
    def calcCloudPriceDivergence(df):
        def _calc(row):
            lead_max = max(row.leadspan_a, row.leadspan_b)
            lead_min = min(row.leadspan_a, row.leadspan_b)
            if lead_max < row.close:
                return (row.close - lead_max) / lead_max
            elif lead_min > row.close:
                return (row.close - lead_max) / lead_max
            else:
                return 0

        arr = df.apply(_calc, axis=1).values
        return arr


# def main():
#     from general import getData
#     df = getData('TSLA', '2016-01-01')
#     # df = calcIchimoku(df)
#     # df = calcRSI(df)
#     df = calcSMA(df)
#     df = calcMACD(df)

#     print(df.tail(20))
# print(calculateData(df))

# if __name__ == '__main__':
#     main()
