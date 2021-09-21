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

df2 = pd.read_csv('Lumpy.csv', names=['Product_Code','ADI','Average','sd','cv_sqr','category'], header=None)
print(df2.head())

lumpy=df.loc[df["Product_Code"].isin(df2["Product_Code"].tolist())]
print(lumpy.head())
lumpy = lumpy.groupby('Date')['Order_Demand'].sum().reset_index()
lumpy=lumpy.set_index('Date')
monthly_avg_sales = lumpy['Order_Demand'].resample('MS').mean()
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
