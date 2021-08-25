import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm, skew #Import Norm and skew for some statistics
from scipy import stats #Import stats
import statsmodels.api as sm #for decomposing the trends, seasonality etc.

from statsmodels.tsa.statespace.sarimax import SARIMAX #for the Seasonal Forecast


#Lets check the ditribution of the target variable (Order_Demand)
from matplotlib import rcParams

df=pd.read_csv('Historical Product Demand.csv')
print(df.dtypes)
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True)


ordinalEncoder = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Product_Code'])
ordinalEncoder.fit(X)
df['Product_Code'] = pd.DataFrame(ordinalEncoder.transform(X))

ordinalEncoder = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Warehouse'])
ordinalEncoder.fit(X)
df['Warehouse'] = pd.DataFrame(ordinalEncoder.transform(X))

ordinalEncoder = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Product_Category'])
ordinalEncoder.fit(X)
df['Product_Category'] = pd.DataFrame(ordinalEncoder.transform(X))
print(df.head(50))

#Since the "()" has been removed , Now i Will change the data type.
df['Order_Demand']=df['Order_Demand'].str.replace('(',"", regex=True)
df['Order_Demand']=df['Order_Demand'].str.replace(')',"", regex=True)
df['Order_Demand'] = df['Order_Demand'].astype('int64')
