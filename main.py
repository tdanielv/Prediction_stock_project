
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Define the stock symbol, the Alpha Vantage API key and the parameters
stock_symbol = "GLD"
api_key = "L58WI1HAC6T3XEKP"
outputsize = 'full'
interval = '60min'

# Initialize the TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

# Get the intraday data
data, meta_data = ts.get_intraday(symbol=stock_symbol, interval=interval)

# Print the data
print(data)
# ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


