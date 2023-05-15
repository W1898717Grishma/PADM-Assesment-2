#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import yfinance as yf

# Set the start and end dates for the data extraction
start_date = '2018-05-10'
end_date = '2023-05-09'

# Extract the historical prices for FTSE 100 from Yahoo Finance
nifty_data = yf.download('^FTSE', start=start_date, end=end_date, interval='1d')

# Calculate the daily returns based on the adjusted closing prices
nifty_data['Daily Returns'] = nifty_data['Adj Close'].pct_change()

# Drop the first row (which contains NaN value) from the dataframe
nifty_data.dropna(inplace=True)

# Filter the data for the last 5 years
daily_returns = nifty_data['Daily Returns'].loc['2018-05-11':]

# Print the last 5 rows of the daily returns data
print(daily_returns.tail())

# Plot a graph of the daily returns over time
daily_returns.plot(figsize=(10,5), title='Daily Returns of FTSE 100 (Last 5 Years)')


# In[14]:


import pandas as pd
import yfinance as yf

# Set the start and end dates for the data extraction
start_date = '2018-05-13'
end_date = '2023-05-13'

# Extract the historical prices for FTSE 100 from Yahoo Finance
nifty_data = yf.download('^FTSE', start=start_date, end=end_date, interval='1d')

# Calculate the daily returns based on the adjusted closing prices
nifty_data['Daily Returns'] = nifty_data['Adj Close'].pct_change()

# Drop the first row (which contains NaN value) from the dataframe
nifty_data.dropna(inplace=True)

# Filter the data for the last 5 years
daily_returns = nifty_data['Daily Returns'].loc['2018-05-14':]

# Print the last 5 rows of the daily returns data
print(daily_returns.tail())

# Plot a graph of the daily returns over time
daily_returns.plot(figsize=(10,5), title='Daily Returns of FTSE 100 (Last 5 Years)')

# Calculate the statistics of the daily returns
stats_df = pd.DataFrame({
    'Mean': daily_returns.mean(),
    'Median': daily_returns.median(),
    'Maximum': daily_returns.max(),
    'Minimum': daily_returns.min(),
    'Std Dev': daily_returns.std(),
    'Skewness': daily_returns.skew(),
    'Kurtosis': daily_returns.kurtosis()
}, index=[0])

print(stats_df)


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Set the start and end dates for the data extraction
start_date = '2018-05-10'
end_date = '2023-05-09'

# Extract the historical prices for FTSE 100 and S&P500 from Yahoo Finance
ftse_data = yf.download('^FTSE', start=start_date, end=end_date, interval='1d')
sp500_data = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')

# Calculate the daily returns based on the adjusted closing prices
ftse_data['Daily Returns'] = ftse_data['Adj Close'].pct_change()
sp500_data['Daily Returns'] = sp500_data['Adj Close'].pct_change()

# Drop the first row (which contains NaN value) from the dataframes
ftse_data.dropna(inplace=True)
sp500_data.dropna(inplace=True)

# Filter the data for the last 5 years
ftse_returns = ftse_data['Daily Returns'].loc['2018-05-16':'2023-05-09']
sp500_returns = sp500_data['Daily Returns'].loc['2018-05-16':'2023-05-09']

# Create two separate subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the FTSE returns on the first subplot
ax1.plot(ftse_returns.index, ftse_returns.values)
ax1.set_title('Daily Returns of FTSE 100 (Last 5 Years)')

# Plot the S&P500 returns on the second subplot
ax2.plot(sp500_returns.index, sp500_returns.values)
ax2.set_title('Daily Returns of S&P 500 (Last 5 Years)')

# Display the plot
plt.show()


# In[6]:


import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

# Set the start and end dates for the data extraction
start_date = '2018-05-10'
end_date = '2023-05-09'

# Extract the historical prices for FTSE 100 from Yahoo Finance
ftse_data = yf.download('^FTSE', start=start_date, end=end_date, interval='1d')

# Calculate the daily returns based on the adjusted closing prices
ftse_data['Daily Returns'] = ftse_data['Adj Close'].pct_change()

# Drop the first row (which contains NaN value) from the dataframe
ftse_data.dropna(inplace=True)

# Filter the data for the last 5 years
daily_returns = ftse_data['Daily Returns'].loc['2018-05-10':]

# Perform the ADF test
result = adfuller(daily_returns)

# Print the test statistic, p-value, and critical values
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[15]:


import matplotlib.pyplot as plt
import datetime as dt  # optional
import arch.data.sp500

st = dt.datetime(2018, 5, 10) # setting the date. Optional line.
en = dt.datetime(2023, 5, 9) # setting the date. Optional line.

import yfinance as yf
data = yf.download("^FTSE", start=st, end=en, adjusted=True)

market = data["Adj Close"]

returns = 100 * market.pct_change().dropna() # generating return series
ax = returns.plot() # ploting the data

#xlim = ax.set_xlim(returns.index.min(), returns.index.max())

# Estimating a GARCH and ARCH models
from arch import arch_model

model1 = arch_model(returns, p=1,o=0,q=0)
out1=model1.fit()
print(out1.summary())

model2 = arch_model(returns, p=1,o=0,q=1)
out2=model2.fit()
print(out2.summary())

fig1 = out1.plot(annualize="D") #generating volatility graphs

fig2 = out2.plot(annualize="D") #generating volatility graphs

# Estimating TGARCH

model3= arch_model(returns, p=1, o=1, q=1, power=1.0)
out3= model3.fit()
print(out3.summary())
fig3 = out3.plot(annualize="D") #generating volatility graphs


# In[16]:


import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

# Load data from yfinance
ticker = '^FTSE'
start_date = '2018-05-10'
end_date = '2023-05-09'
data = yf.download(ticker, start=start_date, end=end_date)['Adj Close'].pct_change().dropna()

# Specify EGARCH model
model = arch_model(data, vol='EGARCH', p=1, o=1, q=1, power=2.0)

# Estimate model
res = model.fit()

# Print model summary
print(res.summary())


# In[ ]:




