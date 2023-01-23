import yfinance as yf
import datetime as dt

# Define the stock symbol and the date range for the data
from matplotlib import pyplot as plt
preicted_price_with_days = {}
for i in range(20):
    days = 10 + 10 * i
    stock_symbol = "GOLD"
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(days))
    # Get the stock data with period of 1 year
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval='1d')
    print(stock_data)

    def show_stock_price():
        stock_data['Adj Close'].plot()
        plt.show()

    def show_stocks_price():
        import pandas as pd
        tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']
        # Import pandas
        data = pd.DataFrame(columns=tickers_list)
        # Fetch the data
        for ticker in tickers_list:
            data[ticker] = yf.download(ticker,start_date,end_date, interval='60m')['Adj Close']
        # Print first 5 rows of the data
        # print(data.head())
        # Plot all the close prices
        ((data.pct_change()+1).cumprod()).plot(figsize=(10, 7))
        # Show the legend
        plt.legend()
        # Define the label for the title of the figure
        plt.title("Adjusted Close Price", fontsize=16)
        # Define the labels for x-axis and y-axis
        plt.ylabel('Price', fontsize=14)
        plt.xlabel('Year', fontsize=14)
        # Plot the grid lines
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.show()
    # show_stocks_price()


    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    # load the stock data
    df = stock_data

    # normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    # create a function to create a data set
    def create_dataset(df):
        x = []
        y = []
        for i in range(len(df)-1):
            x.append(df[i])
            y.append(df[i+1])
        x = np.array(x)
        y = np.array(y)
        return x, y

    # create the data set
    x, y = create_dataset(df)

    # reshape the data
    x = x.reshape(x.shape[0], 1, x.shape[1])

    # create the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dense(1))

    # compile the model
    # the compile() method is used to configure the model's
    # learning process, by specifying the loss function and optimizer to use
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit the model
    # The fit() method is used to train the model on the historical stock data.
    # The epochs parameter is used to specify the number of times the model
    # should iterate over the entire data set
    # and the batch_size parameter is used to specify the number of samples per gradient update
    # The verbose parameter is used to control the amount of information displayed during the training process.
    model.fit(x, y, epochs=100, batch_size=1, verbose=2)

    # make a prediction for the next day
    last_day = df[-1]
    print('----', last_day)
    next_day = model.predict(np.array([last_day]))

    # inverse the normalization
    next_day = scaler.inverse_transform(next_day)

    # print the predicted price for the next day
    print("Predicted price for next day: ", next_day[0][0])
    preicted_price_with_days[days] = next_day[0][0]
    print('------', preicted_price_with_days)