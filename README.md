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
['ExponentialMovingAverage',
 'MovingAverageAnd200SMA',
 'StrategySimple',
 'WeightedMovingAverage',
 'applyStrategiesIchimoku'
 ]
```

Using a strategy
```
from algotrade.strategies import MovingAverageAnd200SMA# import strategy
from ta.trend import ema_indicator # chose indicator from ta library

# read built in doc for __init__ to see available arguments
strategy = MovingAverageAnd200SMA(periods_short=25, periods_long=32, name='ema', indicator=ema_indicator)


test = algotrade.testing.TestStrategy(ticker="AMD", strategy=strategy, start_date='2012-01-01') # Test strategy
print(test) # to print stats

test.plotBuySell(days=500, display_strategy=True) # plot strategy on chart
```

![graph showing price and buy/sell marks](https://raw.githubusercontent.com/t4skmanag3r/algotrade/master/graph.png)
