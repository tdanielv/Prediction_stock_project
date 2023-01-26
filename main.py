import yfinance as yf
import datetime as dt
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
stock_symbol = "GLD"
api_key = "L58WI1HAC6T3XEKP"
outputsize = 'full'
interval = '60min'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_intraday(symbol=stock_symbol, interval=interval)
print(data)

