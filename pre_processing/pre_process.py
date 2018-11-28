# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np
from numpy import log
from pandas import Series
from statsmodels.tsa.stattools import adfuller

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import seaborn as sns
from datetime import datetime
# fix_yahoo_finance is used to fetch data
import fix_yahoo_finance as yf


# Read data
# Df = yf.download('GLD','2008-01-01','2018-10-30')
# Only keep close columns
# Df.to_csv("yahoo_data.csv")


# dates = date2num(df['Name'])
# df['Name'] = [datetime.strptime(i, date_format) for i in df['Name']]
# df['Name'] = dates
# df.to_csv("data_usd.csv")
# Plot the closing price of GLD
# df[ds_gold].plot(figsize=(10, 5))
# plt.ylabel("Gold Pric	es")
# plt.show()
# plt.plot_date(dates, df[ds_gold])
