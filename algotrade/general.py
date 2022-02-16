import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Dict


def get_sp500() -> List[str]:
    """
    Gets all tickers from the S&P500 index fund

    Returns:
        list(str)
    """
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[
        0
    ]
    tickers = tickers.Symbol.to_list()
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers


def get_revolut_stocks() -> List[str]:
    """
    Gets all tickers offered on Revolut trading platform.

    Returns:
        list(str)
    """
    import requests

    req = requests.get("https://globefunder.com/revolut-stocks-list/")
    tickers = list(pd.read_html(req.content)[0]["Symbol"])
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers


def getData(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Gets historical data for ticker from start date to current date.

    Args:
        ticker : str
            ticker to retrieve data for
        start_date : str
            starting date in format "YYYY-MM-DD"

    Returns:
        pandas.DataFrame()
    """
    import yahoo_fin.stock_info as yf

    try:
        df = yf.get_data(ticker, start_date=start_date)
    except Exception as e:
        raise Exception
    if len(df) == 0:
        print("Failed to get data")
    return df


def getBuySellDates(
    df: pd.DataFrame,
    buy_signals: Union[List[bool], np.ndarray],
    sell_signals: Union[List[bool], np.ndarray],
) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Uses logic to invert buy/sell signals and retrieve buy/sell dates

    Args:
        df : pandas.DataFrame()
            Dataframe object of historical data | from general.getData()
        buy_signals : list(bool)
            buying signals retrieved from testing.NewStrategy()[0]
        sell_signals : list(bool)
            selling signals retrieved from testing.NewStrategy()[1]

    Returns:
        buy_dates : list(pandas.Timestamp)
            buying dates
        sell_dates : list(pandas.Timestamp)
            selling dates
    """
    df["buy_signal"], df["sell_signal"] = buy_signals, sell_signals
    buy_dates = []
    sell_dates = []
    buying = False
    first = True
    for i, (_, row) in enumerate(df.iterrows()):
        if row["buy_signal"] == True and buying == False:
            if first != True:
                buying = True
                if len(df) > i + 1:
                    buy_dates.append(row.name)
                else:
                    buy_dates.append(row.name)
        elif row["sell_signal"] == True and buying == True:
            if first != True:
                buying = False
                if len(df) > i + 1:
                    sell_dates.append(row.name)
                else:
                    sell_dates.append(row.name)
        elif row["sell_signal"] == True and buying == False and first == True:
            first = False
    return buy_dates, sell_dates


def calcProfitsWithDate(
    df: pd.DataFrame,
    buy_dates: List[pd.Timestamp],
    sell_dates: List[pd.Timestamp],
) -> List[Tuple[float, pd.Timestamp, pd.Timestamp]]:
    """
    Calculates and returns list of lists that contain [profit, buy_date, sell_date] by subtracting buying from selling date closing prices

    Args:
        df : pandas.DataFrame()
            Dataframe object of historical data | from general.getData()
        buy_dates : list(pandas.Timestamp)
            buying dates | from general.getBuySellDates()[0]
        sell_dates : list(pandas.Timestamp)
            selling dates | from general.getBuySellDates()[1]

    Returns:
        list(list(float, pd.Timestamp, pd.Timestamp))
            list of [profit, buy_date, sell_date]
    """
    profits = []
    for i in range(0, len(buy_dates)):
        if i < len(sell_dates):
            profits.append(
                [
                    (df.loc[sell_dates[i]].close - df.loc[buy_dates[i]].close)
                    / df.loc[buy_dates[i]].close,
                    buy_dates[i],
                    sell_dates[i],
                ]
            )
    return profits


def calcStats(
    profits: List[Tuple[float, pd.Timestamp, pd.Timestamp]]
) -> Dict[str, float]:
    """
    Calculates statistics based on given list of profits

    Args:
        profits : list(list(float, pd.Timestamp, pd.Timestamp))
            list of profits | from general.calcProfitsWithDate()

    Returns:
        dict(str:float)
    """

    profit_vals = [x[0] for x in profits]
    if len(profit_vals) != 0:
        profit_sum = sum(profit_vals) * 100
        profit_mean = np.mean(np.array(profit_vals)) * 100
        profit_median = np.median(np.array(profit_vals)) * 100
        profit_win = len(list(filter(lambda x: x > 0, profit_vals))) / len(profit_vals)
        num_trades = len(profit_vals)
    else:
        profit_sum = 0
        profit_mean = 0
        profit_median = 0
        profit_win = 0
        profit_win = 0
        num_trades = 0
    stats = {
        "profit_sum": profit_sum,
        "profit_mean": profit_mean,
        "profit_median": profit_median,
        "profit_win": profit_win,
        "num_trades": num_trades,
    }
    return stats


def getStrategies() -> List[str]:
    """
    returns a list of avaliable strategies
    """
    import algotrade.strategies as strat

    return [f for f in dir(strat) if not f.startswith("_")]


def __get_and_calculate_data(ticker, start_date):  # deprecate this
    # Gets data and applies calculations
    from algotrade.calculations import calculateData

    df = getData(ticker, start_date=start_date)
    df = calculateData(df)
    return df


def calculateTickersDf(
    ticker_list: List[str], start_date: str
) -> Union[pd.DataFrame, pd.Series]:
    """
    Retrieves and calculates data for multiple tickers in list using multiprocessing, returns merged dataframe

    Args:
        ticker_list : list(str)
            list of tickers to calculate data for
        start_date : str
            starting date for data, format - "YYYY-MM-DD"

    Returns:
        pandas.DataFrame()
            Dataframe of calculated and merged ticker data
    """
    import concurrent.futures

    results = []
    sucesfull = 0
    failed_tickers = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = {
            executor.submit(
                __get_and_calculate_data, ticker, start_date=start_date
            ): ticker
            for ticker in ticker_list
        }
        for future in concurrent.futures.as_completed(res):
            ticker = res[future]
            try:
                data = future.result()
                results.append(data)
                sucesfull += 1
            except Exception as exc:
                failed_tickers.append(ticker)
                print("%r generated an exception: %s" % (ticker, exc))
    return pd.concat(results)


def calculateInvestment(
    timespan_months: int, investment_sum: int, profit_percent: float
):
    """
    Calculates and prints investment statistics for given timespan, investment sum and average profit percentage

    Args:
        timespan_months : int
            timespan in months to calculate investment for
        investment_sum : int
            sum of money invested monthly
        profit_percent : float
            average monthly percent return on investment ex.: 7 %

    Returns:
        None
            Instead prints statistics to console
    """
    invested = investment_sum
    profit_percent = profit_percent / 100
    for month in range(1, timespan_months + 1):
        invested += invested * (profit_percent)
    print(f"invested after timespan: {invested:.2f}$")
    print(f"growth: {(invested - investment_sum):.2f}$")
    print(f"income per month: {(invested * profit_percent):.2f}$")


# def main():
#     # data = getData('TSLA', '2016-01-01')
#     # print(data)

#     # calculateInvestment(12, 1000, 6)

#     df = calculateTickersDf(['TSLA', 'AMD', 'AMAT'], '2016-01-01')
#     df.to_csv('temp.csv', index=0)

# if __name__ == '__main__':
#     main()
