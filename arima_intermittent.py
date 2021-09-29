import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, skew  # Import Norm and skew for some statistics
from scipy import stats  # Import stats
import statsmodels.api as sm  # for decomposing the trends, seasonality etc.
from statsmodels.tsa.statespace.sarimax import SARIMAX  # for the Seasonal Forecast

df = pd.read_csv('Historical Product Demand.csv')
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True)
print(df.head())
df.sort_values('Date')[1:50]
#df['Order_Demand']=df['Order_Demand'].str.replace('(', "")
#df['Order_Demand']=df['Order_Demand'].str.replace(')', "")
df['Order_Demand']=df['Order_Demand'].astype('int64')
df['Date'] = pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year

df2 = pd.read_csv('Intermittent.csv', names=['Product_Code','ADI','Average','sd','cv_sqr','category'], header=None)
print(df2.head())

intm=df.loc[df["Product_Code"].isin(df2["Product_Code"].tolist())]
print(intm.head())
intm = intm.groupby('Date')['Order_Demand'].sum().reset_index()
intm=intm.set_index('Date')
monthly_avg_sales = intm['Order_Demand'].resample('MS').mean()
monthly_avg_sales = monthly_avg_sales.fillna(monthly_avg_sales.bfill())

import itertools
p = d = q = range(0,2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,d,q))]
print('SARIMAX1: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX2: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX3: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX4: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(monthly_avg_sales, order=param, seasonal_order=param_seasonal,enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = sm.tsa.statespace.SARIMAX(monthly_avg_sales, order=(1,1,1),seasonal_order=(0,1,1,12),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=False)
#confidence interval
pred_ci = pred.conf_int()

#plotting real and forecasted values
ax = monthly_avg_sales['2016':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label = 'one-step ahead Forecast', alpha = .7,figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

#Calculating the forecast accuracy
y_forecasted = pred.predicted_mean
y_truth = monthly_avg_sales['2016-01-01':]
mse = ((y_forecasted - y_truth) **2).mean()
print('MSE of Intermittent pattern {}'.format(round(mse,2)))
print('RMSE of Intermittent pattern {}'.format(round(np.sqrt(mse),2)))

pred_uc = results.get_forecast(steps=75)
pred_ci = pred_uc.conf_int()
ax=monthly_avg_sales.plot(label='observed', figsize=(16,8));
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1],color='k',alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()
