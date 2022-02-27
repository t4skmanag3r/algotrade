import pandas as pd
import numpy as np


class NewStrategy:
    """
    Create a new strategy based on strategies.combineStrategies(), used for constructing a strategy before running the calculation itself
      Args:
        buy_strategies : list(str)
            Use strategies.strategyList() to get possible strategies
        buy_weights : list(int)
            corespoding weights to given strategies
        sell_strategies : list(str)
            Use strategies.strategyList() to get possible strategies
        sell_weights : list(int)
            corespoding weights to given strategies (non negative)
        buy_threshold : int
            threshold needed to triger buy signal
        buy_threshold : int
            threshold needed to triger sell signal (non negative)

        Returns:
            <NewStrategy> object which on call uses args on strategies.combinedStrategies()
    """

    def __init__(
        self,
        buy_strategies,
        buy_weights,
        sell_strategies,
        sell_weights,
        buy_threshold,
        sell_threshold,
    ):
        self.buy_strategies = buy_strategies
        self.buy_weights = buy_weights
        self.sell_strategies = sell_strategies
        self.sell_weights = sell_weights
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def __repr__(self):
        return f"""
Strategy: {self.__class__}
buy_strategies = {self.buy_strategies}
buy_weights = {self.buy_weights}
sell_strategies = {self.sell_strategies}
sell_weights = {self.sell_weights}
buy_threshold = {self.buy_threshold}
sell_threshold = {self.sell_threshold}
        """

    def __call__(self, df):
        from algotrade.strategies import combineStrategies

        return combineStrategies(
            df,
            self.buy_strategies,
            self.buy_weights,
            self.sell_strategies,
            self.sell_weights,
            self.buy_threshold,
            self.sell_threshold,
        )


# different implimentation with weights and biases
# class NewStrategy:
#     def __init__(self, strategies, weights, biases, buy_threshold, sell_threshold):
#         self.strategies = strategies
#         self.weights = weights
#         self.biases = biases
#         self.buy_threshold = buy_threshold
#         self.sell_threshold = sell_threshold

#     def __repr__(self):
#         return f"""
# Strategy: {self.__class__}
# strategies = {self.strategies}
# weights = {self.weights}
# biases = {self.biases}
# buy_threshold = {self.buy_threshold}
# sell_threshold = {self.sell_threshold}
#         """

#     def __call__(self, df):
#         from algotrade.strategies import combineStrategies
#         return combineStrategies(df, self.strategies, self.weights, self.biases, self.buy_threshold, self.sell_threshold)


class TestStrategy:
    """
    Class for testing strategies

    Attributes:
        ticker : str
            ticker to test strategy on
        start_date : str
            date to start testing from, format-"YYYY-MM-DD"
        strategy : <NewStrategy>
            class NewStrategy with set arguments to test on
        df : pandas.DataFrame()
            dataframe of historical ticker data

    Methods:
        calcStats()
            calculates strategy performance statistics
        plotBuySell(days, display_ichimoku=False, trade_lines=False)
            plots historical price data for ticker with matplotlib
        plotMACD(days):
            plots MACD graph
        plotEMA(days):
            plots EMA graph
    """

    def __init__(self, ticker: str, strategy, start_date="2012-01-01", df=None):
        """
        Args:
            ticker : str
                ticker to test strategy on
            start_date : str
                date to start testing from, format-"YYYY-MM-DD"
            strategy : <NewStrategy>
                class NewStrategy with set arguments to test on
            df : pandas.DataFrame()
                dataframe of historical ticker data
        """
        self.ticker = ticker
        self.start_date = start_date
        self.strategy = strategy
        self.df = df
        self.calcStats()

    def __repr__(self):
        return f"""
ticker = {self.ticker}
start_date = {self.start_date}
strategy = {type(self.strategy)}
stats = {self.stats}
        """

    def calcStats(self):
        """
        Calculates performance statistics
        """
        from algotrade.general import (
            getData,
            getBuySellDates,
            calcProfitsWithDate,
            calcStats,
        )
        from algotrade.calculations import calculateData

        if isinstance(self.df, pd.DataFrame):
            try:
                self.df = self.df.loc[self.df["ticker"] == self.ticker].copy()
            except Exception as e:
                raise e
        else:
            print("Retrieving data")
            self.df = getData(self.ticker, self.start_date)
            # self.df = calculateData(self.df)

        buy_signals, sell_signals = self.strategy.apply(df=self.df)
        self.buy_dates, self.sell_dates = getBuySellDates(
            self.df, buy_signals, sell_signals
        )
        self.profits = calcProfitsWithDate(self.df, self.buy_dates, self.sell_dates)
        self.stats = calcStats(self.profits)

    def plotProfitDistribution(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 6))
        plt.hist([i[0] * 100 for i in self.profits], bins="doane")
        plt.xlabel("Profit %")

        return plt.show()

    def plotBuySell(
        self, days=None, display_strategy=False, display={}, scale_log=False
    ):
        """
        Plots historical price data for ticker with matplotlib

        Args:
            days : int
                number of days to plot data for
            display_strategy : bool
                displays technical indicators used to make make signals for strategy
            display : dict{str : list[int]/None}
                dictionary of keys(str) from technical indicators from algotrade.ploting - [
                    sma,
                    ema,
                    ichimoku,
                    macd
                ]
                and values(list[int] or None if deffaults) timeframes to calculate indicator to display
                examples: {"sma" : [50, 100, 200]} - this calculates the simple moving average of 50, 100, 200 days and displays it,
                {"ichimoku" : None} - shows all ichimoku indicators
            scale_log : bool
                sets graph scaling method to logarithmic
        """
        # setting up figure
        import matplotlib.pyplot as plt

        f1 = plt.figure(figsize=(24, 9))
        ax1 = f1.add_subplot(111)

        # Day checking, filtering
        if days is None:
            df = self.df
        elif days < len(self.df):
            df = self.df[-days:]
        else:
            df = self.df

        buy_dates = [date for date in self.buy_dates if date in df.index.values]
        sell_dates = [date for date in self.sell_dates if date in df.index.values]
        profits = [x[0] * 100 for x in self.profits]
        profits = profits[(len(profits) - len(sell_dates)) :]
        # print(f"buy dates:\n{buy_dates},\nsell_dates:\n{sell_dates}")

        # ploting price
        ax1.plot(df.index, df.close, color="gray")
        legend = ["price"]

        # ploting strategy indicators
        if display_strategy:
            self.strategy.plot(days)
            legend = legend + self.strategy.legend

        # display dictionary checking and ploting indicators from ploting
        if len(display.items()) > 0:
            for disp, val in display.items():
                if type(val) is list:
                    for v in val:
                        indicator = disp(df, v)
                        legend = legend + indicator.legend
                elif val is None:
                    indicator = disp(df)
                    legend = legend + indicator.legend
                else:
                    raise TypeError

        # ploting buy/sell signals
        ax1.scatter(
            df.loc[buy_dates].index, df.loc[buy_dates]["adjclose"], marker="^", c="g"
        )
        ax1.scatter(
            df.loc[sell_dates].index, df.loc[sell_dates]["adjclose"], marker="v", c="r"
        )

        # displaying signal profit text
        for i, s in enumerate(sell_dates):
            if i < len(sell_dates):
                ax1.text(
                    df.loc[s].name,
                    df.loc[s]["adjclose"] - (df["close"].max() / 100),
                    s=f"{profits[i]:.2f}%",
                    fontdict=dict(size=12),
                )

        # showing legend and labels
        ax1.legend(legend)
        plt.xlabel("Date")
        plt.ylabel("Price")

        # log scale graph
        if scale_log:
            ax1.yscale("log")

        # autoscaling
        ax1.autoscale(axis="y")

        return plt.show()


def testStrategyMultipleStocks(ticker_list, strategy, start_date="2012-01-01", df=None):
    """
    Tests strategy on multiple tickers and returns dataframe of performance for each ticker

    Args:
        ticker_list : list(str)
            list of tickers to test on
        strategy : <NewStrategy>
            class NewStrategy with arguments to test performance of
        start_date : str
            date to test from, format-"YYYY-MM-DD"

    Returns:
        pd.DataFrame()
            dataframe of perormance data for each ticker
    """
    import concurrent.futures

    results = []
    sucesfull = 0
    failed_tickers = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = {
            executor.submit(
                TestStrategy, ticker, strategy=strategy, start_date=start_date, df=df
            ): ticker
            for ticker in ticker_list
        }
        for future in concurrent.futures.as_completed(res):
            ticker = res[future]
            try:
                data = future.result()
                if data != None:
                    stats = data.stats
                    stats["ticker"] = ticker
                    results.append(stats)
                    sucesfull += 1
            except Exception as exc:
                failed_tickers.append(ticker)
                print("%r generated an exception: %s" % (ticker, exc))
    df = pd.DataFrame.from_dict(results)
    df = df.set_index("ticker")
    return df


# def main():
# from strategies import strategy_MA_MACD
# buy_strategies = ["buy_conv", "buy_cloud", "buy_leadspan"]
# buy_weights = [1, 4, 3]
# buy_weights = [2, 1, 3]
# sell_strategies = ["sell_conv", "sell_cloud", "sell_leadspan"]
# sell_weights = [1, 3, 1]
# sell_weights = [1, 1, 1]
# buy_threshold = 4
# sell_threshold = 3
# buy_threshold = 0
# sell_threshold = 0
# strategy = NewStrategy(buy_strategies, buy_weights, sell_strategies, sell_weights, buy_threshold, sell_threshold)

# strategy = strategy_MA_MACD
# testStrat = TestStrategy('AAPL', '2012-01-01', strategy)
# stats = testStrat.stats
# from pprint import pprint
# pprint(stats)
# testStrat.plotBuySell(1000, display_ichimoku=False, display_ma=True)

# testStrat.plotMACD(90)
# testStrat.plotEMA(90)

# from general import get_sp500, get_revolut_stocks
# # tickers = get_sp500()
# tickers = get_revolut_stocks()
# df = pd.read_csv('temp.csv')
# tickers = list(df.ticker.unique())
# res = testStrategyMultipleStocks(tickers, strategy)
# res.to_csv('stats_macd.csv')
# print(res)

# from strategies import (
#     StrategySimple,
#     WeightedMovingAverage,
#     ExponentialMovingAverage,
#     SMA200,
#     MACD,
# )
# import ploting
# from general import getData
# from ta.trend import ema_indicator

# ticker = "AMAT"
# data = getData(ticker, "2012-01-01")

# class NewStrat(SMA200):
#     def __init__(
#         self,
#         periods_short=25,
#         periods_long=32,
#         name="ema",
#     ):
#         super().__init__(
#             indicator=ema_indicator,
#             periods_short=periods_short,
#             periods_long=periods_long,
#             name=name,
#         )

# strategy = MACD()
# test = TestStrategy(ticker, strategy, df=data)
# print(test)

# test.plotBuySell(scale_log=False, display_strategy=True, days=500)

# from tqdm import tqdm

# results = pd.DataFrame(
#     columns=[
#         "short",
#         "long",
#         "profit_sum",
#         "profit_mean",
#         "profit_median",
#         "profit_win",
#         "num_trades",
#     ]
# )

# for short in tqdm(range(3, 60, 2)):
#     for long in tqdm(range(short + 2, 80, 2)):
#         strategy = SMA200(
#             indicator=ema_indicator,
#             periods_short=short,
#             periods_long=long,
#             name="ema",
#         )
#         test = TestStrategy(ticker, strategy, df=data)
#         results = pd.concat(
#             [
#                 results,
#                 pd.DataFrame.from_dict(
#                     {"short": [short], "long": [long], **test.stats},
#                 ),
#             ],
#             ignore_index=True,
#         )

# results.to_csv("results.csv")

# test.plotProfitDistribution()
# test.plotBuySell(scale_log=False, display_strategy=True)
# test.plotBuySell(
#     scale_log=False,
#     days=500,
#     display_strategy=False,
#     display={ploting.Ema: [200, 100], ploting.Ichimoku: None, ploting.Macd: None},
# )

# import ploting
# ploting.macd(data)

# test.plotBuySell(display={'sma':[200]})
# test.plotBuySell(display={'sma':[50, 100, 200]})


# if __name__ == "__main__":
#     main()
