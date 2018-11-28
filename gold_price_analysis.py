# -*- coding: utf-8 -*-

# from google.colab import drive
# drive.mount("/content/drive")

# import os
# os.chdir("/content/drive/My Drive/Colab Notebooks/data")
# print(os.getcwd())
# Change directory to the directory above "data"

# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np
from math import sqrt
from numpy import log
from pandas import Series



from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels as sm

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import seaborn as sns
from datetime import datetime
import subprocess
# fix_yahoo_finance is used to fetch data
# import fix_yahoo_finance as yf

ds_gold = 'Indian rupee'
ds_etf = 'Close'
date_format = '%Y-%m-%d'
df = pd.read_csv("data_inr.csv")
df = df[['Name', ds_gold]]
df['Name'] = [datetime.strptime(i, date_format) for i in df['Name']]
df.set_index('Name')
# df.index = pd.to_datetime(df.index, format=date_format)
print(df.columns)
dd =df

"""*  Drop rows with missing values"""

df = df.dropna()

df[ds_gold].hist()
plt.show()
log_transform = log(df[ds_gold])
print(min(log_transform), max(log_transform))

sns.set()
sns.distplot(df[ds_gold])
plt.show()

plt.plot(df['Name'], df[ds_gold])
plt.show()

plt.plot(df['Name'], log_transform)
plt.show()

# Can be used to show non stationary

# Define exploratory variables
# Finding moving average of past 3 days and 9 days
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
df = df.dropna()
X = df[['S_1', 'S_2']]
X.head()
plt.plot(df['Name'], df['S_1'])
plt.plot(df['Name'], df["S_2"])
plt.show()


# dependent variable
y = df[ds_gold]
y.head()

# Split into train and test
t = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=t, shuffle=False)

# Performing linear regression
linear = LinearRegression().fit(X_train, y_train)

print("Gold Price =", round(linear.coef_[0], 2), "* 2 Month Moving Average", round(
    linear.coef_[1], 2), "* 1 Month Moving Average +", round(linear.intercept_, 2))

# Predict prices
predicted_price = linear.predict(X_test)

predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold Price")
plt.show()

# Calculate R square and rmse to check goodness of fit
r2_score = linear.score(X_test, y_test)*100
print("R square for regression", float("{0:.2f}".format(r2_score)))
sqrt(mean_squared_error(y_test,predicted_price))

# We observe significantly different accuracies for same dataset in USD and INR.
# The reason for this difference could be attributed to the

# Check stationarity
X = df[ds_gold]
split = len(X) // 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

result_of_adfuller = adfuller(df[ds_gold])
print('ADF Statistic: %f' % result_of_adfuller[0])
print('p-value: %f' % result_of_adfuller[1])
print('Critical Values:')
for key, value in result_of_adfuller[4].items():
    print('\t%s: %.3f' % (key, value))

# we can conclude it has time dependent structure and cannot reject null hypothesis.

# from statsmodels.tsa.seasonal import seasonal_decompose
# print(df.index.dtype)
# print(df[ds_gold].dtype)
# decomposition = seasonal_decompose(df[ds_gold], freq = 200)

# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid

# plt.subplot(411)
# plt.plot(df[ds_gold], label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()

# print(df.head())

# Now taking log transform
log_transform = log(df[ds_gold])
result_of_adfuller = adfuller(log_transform)
print('ADF Statistic: %f' % result_of_adfuller[0])
print('p-value: %f' % result_of_adfuller[1])
print('Critical Values:')
for key, value in result_of_adfuller[4].items():
    print('\t%s: %.3f' % (key, value))

# To remove trends, differencing of order 1
k = df[ds_gold].diff()
plt.plot(df['Name'], k)
plt.show()
# print(k.head())
k = k.dropna()

# check stationarity after differencing
result_of_adfuller = adfuller(k)
print('ADF Statistic: %f' % result_of_adfuller[0])
print('p-value: %f' % result_of_adfuller[1])
print('Critical Values:')
for key, value in result_of_adfuller[4].items():
    print('\t%s: %.3f' % (key, value))


# So now we can say with 1 % confidence level that its stationary
# We can do other stuff now

# Again regression
df[ds_gold] = k
# Finding moving average of past 3 days and 9 days
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
df = df.dropna()
X = df[['S_1', 'S_2']]
X.head()
print(X.head())
plt.plot(df['Name'], df['S_1'])
plt.plot(df['Name'], df["S_2"])
plt.show()

df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()

# dependent variable
y = df[ds_gold]
y.head()
# print(y.head())

# Split into train and test
t = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=t, shuffle=False)

# Performing linear regression
linear = LinearRegression().fit(X_train, y_train)

print("Gold Price =", round(linear.coef_[0], 2), "* 2 Month Moving Average", round(
    linear.coef_[1], 2), "* 1 Month Moving Average +", round(linear.intercept_, 2))

# Predict prices
predicted_price = linear.predict(X_test)

predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold Price")
plt.show()

# Calculate R square and rmse to check goodness of fit
r2_score = linear.score(X_test, y_test)*100
print("R square for regression", float("{0:.2f}".format(r2_score)))
sqrt(mean_squared_error(y_test,predicted_price))

# Trying 2nd order differencing
k = df[ds_gold].diff().diff()
plt.plot(df['Name'], k)
plt.show()
# print(k.head())
k = k.dropna()

# check stationarity after differencing
result_of_adfuller = adfuller(k)
print('ADF Statistic: %f' % result_of_adfuller[0])
print('p-value: %f' % result_of_adfuller[1])
print('Critical Values:')
for key, value in result_of_adfuller[4].items():
    print('\t%s: %.3f' % (key, value))

# Again regression
df[ds_gold] = k
# Finding moving average of past 3 days and 9 days
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
df = df.dropna()
X = df[['S_1', 'S_2']]
X.head()
print(X.head())
plt.plot(df['Name'], df['S_1'])
plt.plot(df['Name'], df["S_2"])
plt.show()


# dependent variable
y = df[ds_gold]
y.head()
# print(y.head())
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
# Split into train and test
t = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=t, shuffle=False)

# Performing linear regression
linear = LinearRegression().fit(X_train, y_train)

print("Gold Price =", round(linear.coef_[0], 2), "* 2 Month Moving Average", round(
    linear.coef_[1], 2), "* 1 Month Moving Average +", round(linear.intercept_, 2))

# Predict prices
predicted_price = linear.predict(X_test)

predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold Price")
plt.show()

# Calculate R square and rmse to check goodness of fit
r2_score = linear.score(X_test, y_test)*100
print("R square for regression", float("{0:.2f}".format(r2_score)))
print("RMSE: ",sqrt(mean_squared_error(y_test,predicted_price)))

#Now after 2nd order differencing the results make sense, but are still unacceptably innacurate

#Let us now try to make an ARMA model with the new non-stationary data

# order = arma_order_select_ic(df[ds_gold])
# print(df[ds_gold])

#ACF and PACF plots

series = df[ds_gold]
plt.figure()
plt.subplot(211)
plot_acf(series, ax=plt.gca())
plt.subplot(212)
plot_pacf(series, ax=plt.gca())
plt.show()

del df['S_1']
del df['S_2']

# data = pd.Series(df['Indian rupee'], index=df['Name'])
# model = ARMA(data, order=(5,1))
# data
# df
# arma_model = ARMA(df,order = (2,3))

# df['Name'] = df['Name'].values.astype(float)
# ts = pd.Series(df[ds_gold], index = df.index)
# print(ts.head())
# model = ARIMA(df[ds_gold].values, order=(1, 1, 1))  
# results_ARIMA = model.fit(disp=-1)  
# # np.asarray(dd)
# plt.plot(df)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

# from statsmodels.tsa.statespace import SARIMAX
import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(df[ds_gold].values,order=(2, 1, 2),seasonal_order=(2, 1, 2, 12),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()

results.summary()
df['sarimax_predict'] = results.predict()

# del df['S_1']
# del df['S_2']
df.plot(y = ['Indian rupee','sarimax_predict'], x = df['Name'])

results.summary()

results.plot_diagnostics(figsize=(15, 12))
plt.show()

print("RMSE: ",sqrt(mean_squared_error(df[ds_gold],df['sarimax_predict'])))
from sklearn.metrics import r2_score
print("R2 SCORE: ",r2_score(df[ds_gold],df['sarimax_predict']))



#we have finally reached a good model. GG

#Finding trends

import datetime as dt

data = pd.read_csv("dataset2018.csv") 
data.head()
x_18 = data.iloc[:,0]
y_18 = data.iloc[:,1]
new_18 = [dt.datetime.strptime(d,'%d-%m-%Y').date() for d in x_18]
plt.plot(new_18,y_18, '.r',color='g')
plt.xlabel('date')
plt.ylabel('price')
plt.title('gold price in the year 2018')
plt.show()

data2017 = pd.read_csv("dataset2017.csv") 
x_17 = data2017.iloc[:,0]
y_17 = data2017.iloc[:,1]
new_17 = [dt.datetime.strptime(d,'%d-%m-%Y').date() for d in x_17]
plt.plot(new_17,y_17, '.r',color='g')
plt.xlabel('date')
plt.ylabel('price')
plt.title('gold price in the year 2017')
plt.show()

data2016 = pd.read_csv("dataset2016.csv") 
x_16 = data2016.iloc[:,0]
y_16 = data2016.iloc[:,1]
new_16 = [dt.datetime.strptime(d,'%d-%m-%Y').date() for d in x_16]
plt.plot(new_16,y_16, '.r',color='g')
plt.xlabel('date')
plt.ylabel('price')
plt.title('gold price in the year 2016')
plt.show()

data2015 = pd.read_csv("dataset2015.csv")
x_15 = data2015.iloc[:,0]
y_15 = data2015.iloc[:,1]
new_15 = [dt.datetime.strptime(d,'%d-%m-%Y').date() for d in x_15]
plt.plot(new_15,y_15, '.r',color='g')
plt.xlabel('date')
plt.ylabel('price')
plt.title('gold price in the year 2015')
plt.show()

data2014 = pd.read_csv("dataset2014.csv")
x_14 = data2014.iloc[:,0]
y_14 = data2014.iloc[:,1]
new_14 = [dt.datetime.strptime(d,'%d-%m-%Y').date() for d in x_14]
plt.plot(new_14,y_14, '.r',color='g')
plt.xlabel('date')
plt.ylabel('price')
plt.title('gold price in the year 2014')
plt.show()

data2014 = pd.read_csv("dataset2013.csv")
x_14 = data2014.iloc[:,0]
y_14 = data2014.iloc[:,1]
new_14 = [dt.datetime.strptime(d,'%d-%m-%Y').date() for d in x_14]
plt.plot(new_14,y_14, '.r',color='g')
plt.xlabel('date')
plt.ylabel('price')
plt.title('gold price in the year 2013')
plt.show()

data2014 = pd.read_csv("datasetfull.csv")
x_14 = data2014.iloc[:,0]
y_14 = data2014.iloc[:,1]
new_14 = [dt.datetime.strptime(d,'%d-%m-%Y').date() for d in x_14]
plt.plot(new_14,y_14, '.r',color='g')
plt.xlabel('date')
plt.ylabel('price')
plt.title('gold price')
plt.show()

