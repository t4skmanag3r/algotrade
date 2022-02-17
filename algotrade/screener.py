import pandas as pd
import numpy as np


def __get_result_buy(
    ticker, start_date, last_days, strategy
):  # function to screen last day of stock and test peformance
    from datetime import datetime
    from algotrade.testing import TestStrategy

    today = datetime.today()
    try:
        test = TestStrategy(ticker, start_date, strategy)
    except Exception as e:
        print(f"{ticker} : failed with exception [{e}]")
        return None
    if len(test.buy_dates) != 0:
        if (today - test.buy_dates[-1].to_pydatetime()).days < last_days:
            stats = TestStrategy(ticker, "2012-01-01", strategy).stats
            return [ticker, test.buy_dates[-1], stats]
    return None


def screenStocks_Multiproccesed(ticker_list, strategy, last_days):
    """
    Screens stock list with strategy. Returns list of [tickers, last buy dates, stock performance stats] if last buy date of stock is within last_days
    This particular function is multiproccesed with pythons concurrent.futures package

    Args:
        ticker_list : list(str)
            list of tickers to screen stocks on
        strategy : <NewStrategy>
            class strategies.NewStrategy() with arguments to scren with
        last_days : int
            maximum days trade had to occur in the past for screener to detect

    Returns:
        list(list)
            list of lists that contain [ticker, last buy date, stock performance stats]
    """
    import concurrent.futures
    import datetime

    results = []
    sucesfull = 0
    failed_tickers = []
    start_date = (datetime.datetime.today() - datetime.timedelta(days=1000)).strftime(
        "%Y-%m-%d"
    )

    print("Starting screener...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        dates = {
            executor.submit(
                __get_result_buy,
                ticker,
                strategy=strategy,
                start_date=start_date,
                last_days=last_days,
            ): ticker
            for ticker in ticker_list
        }
        for future in concurrent.futures.as_completed(dates):
            ticker = dates[future]
            try:
                data = future.result()
                if data != None:
                    results.append(data)
                    print(data)
                    sucesfull += 1
            except Exception as exc:
                failed_tickers.append(ticker)
                print("%r generated an exception: %s" % (ticker, exc))
    return results


def screenStocks(ticker_list, strategy, last_days):
    """
    Screens stock list with strategy. Returns list of [tickers, last buy dates, stock performance stats] if last buy date of stock is within last_days
    This particular function is multiproccesed with pythons concurrent.futures package
    Not multiproccesed version of screenStocks_Multiproccesed function

    Args:
        ticker_list : list(str)
            list of tickers to screen stocks on
        strategy : <NewStrategy>
            class strategies.NewStrategy() with arguments to scren with
        last_days : int
            maximum days trade had to occur in the past for screener to detect

    Returns:
        list(list)
            list of lists that contain [ticker, last buy date, stock performance stats]
    """
    import datetime

    start_date = (datetime.datetime.today() - datetime.timedelta(days=1000)).strftime(
        "%Y-%m-%d"
    )
    stocks_to_buy = []
    for ticker in ticker_list:
        test = __get_result_buy(ticker, start_date, last_days, strategy)
        if test != None:
            stocks_to_buy.append(test)
            print(test)
    return stocks_to_buy


def main():
    from general import get_sp500, get_revolut_stocks
    from testing import NewStrategy

    # from strategies import strategy_MA_MACD

    buy_strategies = ["buy_conv", "buy_cloud", "buy_leadspan"]
    buy_weights = [1, 4, 3]
    sell_strategies = ["sell_conv", "sell_cloud", "sell_leadspan"]
    sell_weights = [1, 3, 1]
    buy_threshold = 4
    sell_threshold = 3
    strategy = NewStrategy(
        buy_strategies,
        buy_weights,
        sell_strategies,
        sell_weights,
        buy_threshold,
        sell_threshold,
    )
    # strategy = strategy_MA_MACD

    # tickers = get_sp500()
    tickers = get_revolut_stocks()
    # tickers = ['AMD', 'AMAT', 'AAPL']
    stocks_to_buy = screenStocks_Multiproccesed(tickers, strategy, 7)
    print("-" * 100)
    for stock in sorted(
        stocks_to_buy, key=lambda x: x[1].to_pydatetime(), reverse=True
    ):
        print(stock)

    # x = __get_result_buy('AAPL', '2020-01-01', 14, strategy)
    # print(x)


if __name__ == "__main__":
    main()
