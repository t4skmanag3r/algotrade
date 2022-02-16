# Entire file can be deprecated after removal from general.py dependencies
# since indicators are calculated with external ta library

import pandas as pd
import numpy as np


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
