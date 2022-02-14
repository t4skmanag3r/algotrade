# algotrade

algotrade is a code library that helps to create and test algorithmic trading strategies

Developed by Edvinas Adomaitis (c) 2021

## Examples of How To Use (Alpha Version)
Runnable jupyter notebook also added - example.ipynb

### Creating and testing custom strategy

Import algotrade
```
import algotrade
```

Checking available strategies
```
print(algotrade.strategies.strategyList())
```
output:
```
['buy_conv',
 'sell_conv',
 'buy_cloud',
 'sell_cloud',
 'buy_leadspan',
 'sell_leadspan',
 'Buy_MA_MACD',
 'Sell_MA_MACD']
```

Defining new strategy
```
buy_strategies = ["buy_conv", "buy_cloud", "buy_leadspan"]
buy_weights = [1, 4, 3]
sell_strategies = ["sell_conv", "sell_cloud", "sell_leadspan"]
sell_weights = [1, 3, 1]

buy_threshold = 4
sell_threshold = 3

strategy = algotrade.testing.NewStrategy(buy_strategies, buy_weights, sell_strategies, sell_weights, buy_threshold, sell_threshold)
```

Initializing testing
```
ticker = 'AAPL'
start_date = '2012-01-01'

test = algotrade.testing.TestStrategy(ticker, start_date, strategy)
```

Printing statistics
```
print(test.stats)
```
output:
```
{'profit_sum': 211.06937710317158,
 'profit_mean': 26.383672137896454,
 'profit_median': 30.157999645678284,
 'profit_win': 0.75,
 'num_trades': 8}
```

Ploting buy/sell graph
```
days = 1000
display_ichimoku = False
display_ma = True

test.plotBuySell(days, display_ichimoku, display_ma)
```
![graph showing price and buy/sell marks](https://raw.githubusercontent.com/t4skmanag3r/algotrade/master/graph.png)


## To do:

* create sepperate plotting.py module which has all the plotting functions, use seaborn or plotly dash
* move plotBuySell() and make it usable with either inputs(df, buy_dates, sell_dates) or inputs(class TestStrategy)
