import numpy as np
import pandas as pd
import requests
class Stocks(object):
    def __init__(self, symbol):
        super(Stocks, self).__init__()
        self.symbol = symbol
    def info(self):
        data_info = requests.get("https://finfo-api.vndirect.com.vn/stocks").json()['data']
        for data in data_info:
            if data['symbol'].lower() == self.symbol:
                return data
        return 'Not INFO'
    def events(self):
        data_event = requests.get("https://finfo-api.vndirect.com.vn/events?symbols={}".format(self.symbol.lower())).json()['data']
        return pd.DataFrame.from_dict(data_event)
    def prices(self, start, end):
        start = pd.Timestamp(start).value // (10**9)
        end = pd.Timestamp(end).value // (10**9)
        data_prices = requests.get(f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={self.symbol}&from={start}&to={end}").json()
        df_prices = pd.DataFrame.from_dict(data_prices)
        df_prices.rename(columns={
                't': 'date'
                ,'o': 'open' ,'h': 'high'
                ,'l': 'low'  ,'c': 'close'
                ,'v': 'volume'}
            , inplace=True)
        df_prices['date'] = pd.to_datetime(df_prices['date'], unit='s')
        df_prices.drop(columns=['s'], errors='ignore', inplace=True)

        return df_prices
