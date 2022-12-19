# GOLD PRICE ANALYSIS AND PREDICTION

### DATASET:

 We have used 2 datasets for this. The first data set has been gotten from the “World
 Gold Council’s” website. This dataset is for the global gold prices from 1978 to 2018.
 This is monthly data.

 Since this dataset is not the actual market price of gold, we decided to get the market
 price of gold in India. There were no datasets available for the same. We scraped the
 market data from ​ Gold Price India​ from 2011 - 2018 for every month.
 Since the data was scraped from the web, we verified it manually before using the same
 for analytics.

 ### PREPROCESSING:

 We preprocess all the data by using basic techniques like dropping all the rows with
 missing values (there weren’t any!). The most preprocessing was done to the dates,
 since the data was collected from different sources, the dates were in different formats
 and they were formatted to a common format which would be understood by
 matplotlib for plotting it appropriately.

 ### CREATING A MODEL:

 ● Multiple Regression

 We try to create a simple regression model. It is a multiple regression model with
 input parameters as the moving average of the past 1 month and the past 2
 months.
 We can clearly observe overfitting in this model.This overfitting can be attributed to the data being non-stationary.

 ● Check Stationarity

 To check the stationarity of the data we plot the data along with the dates.
 Just by looking at the plot we can conclude that the data is non-stationary.
 We can also see from the histograms(in the code) that the data has seasonality
 and some component of trends.
 We also performed the dickey-fuller test to confirm the stationarity.
 We can see the ADF statistic is higher than any of the critical values, and the p
 value is much greater than 0.05, so we cannot reject the null hypothesis that the data is
 non-stationary.

 ● Make Data StationaryTo make the data stationary, we use simplest technique of taking a log transform.

 We can observe that there is no change and the data is still non-stationary.
 We now try to perform differencing on this data.
 We perform differencing of order 2 and observe the following results.We can see that the ADF statistic is less than 1% critical value hence we can reject the
 null hypothesis and conclude with a confidence level of 99% that the data is stationary.
 We can now use this data for further modelling.

 ● Regression Model Again

 We use the old regression model again for this stationary data. We see the following
 results.We obtain R square value of 30% which is below par. And the root mean square error is
 also very high. Though the RMSE is an absolute statistic and it cannot be used to judge
 the goodness of fit, we will use this value for further comparison with other models

 ● ACF and PACF Plots

 We now try to plot ACF i.e Autocorrelation and PACF i.e. Partial Autocorrelation plots
 for this data to find the p, q, d values for creating an ARIMA model.

 ● SARIMA model

 We model this data using a SARIMA model. A SARIMA model stands for a
 Seasonal ARIMA model. SARIMA Model is better over a simple ARIMA model
 when there is seasonal data. I.e. the timeseries data has repeating cycles.We observe that the model fits much better than any of the previous models.

 Below are the results and the diagnostics of the model.We see that the R square value is 73% which is acceptable and the RMS Error has
 reduced to 1715 from 5000, which is a good sign.

 ### FINDING TRENDS:

 We now use the other (Indian market dataset) for trying and finding any
 interesting trends in the price fluctuation.Interestingly for various years the prices of gold have been maximum during the
 wedding season corresponding to that year. We can also see a cyclic trend in the price, there is 6-8 years of bullish growth followed by 6-8 years of bearish market. Alhough other than that there are no obvious trends in the
 data. The maximum price has always been March-April or September-October which
 falls during or just before the wedding season.


 ### OBSERVATIONS and CONCLUSION:

 We now have a model which can predict the gold price with almost 73% accuracy and
 have found an interesting correlation between the market price of gold and the
 wedding season in India.

 We can safely conclude that the price of gold in the world market and the regional
 Indian market are very volatile and depend a lot of external factors which cannot be
 modelled so easily.

 For future work, we can use and build upon our existing model to build a
 recommendation system suggesting the users the right time to buy and sell gold for
 people who take interest in investing in gold.

 ## Credits

 * [Nihali S Shetty](https://github.com/NihaliShetty)
 * [Mayank Thakur](https://github.com/IamMayankThakur)
