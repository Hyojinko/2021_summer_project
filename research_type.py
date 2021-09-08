
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import seaborn as sn
from sklearn.preprocessing import StandardScaler


from scipy.stats import norm, skew  # Import Norm and skew for some statistics
from scipy import stats  # Import stats
import statsmodels.api as sm  # for decomposing the trends, seasonality etc.

from statsmodels.tsa.statespace.sarimax import SARIMAX  # for the Seasonal Forecast

# Lets check the ditribution of the target variable (Order_Demand)
from matplotlib import rcParams

df = pd.read_csv('Historical Product Demand.csv')
print(df.dtypes)
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True)


def ordinalEncode_category(df, str):
    ordinalEncoder = preprocessing.OrdinalEncoder()
    X = pd.DataFrame(df[str])
    ordinalEncoder.fit(X)
    df[str] = pd.DataFrame(ordinalEncoder.transform(X))





ordinalEncode_category(df, "Product_Code")
ordinalEncode_category(df, "Warehouse")
ordinalEncode_category(df, "Product_Category")
ordinalEncode_category(df, "Date")

# Since the "()" has been removed , Now i Will change the data type.
df['Order_Demand'] = df['Order_Demand'].str.replace('(', "", regex=True)
df['Order_Demand'] = df['Order_Demand'].str.replace(')', "", regex=True)
df['Order_Demand'] = df['Order_Demand'].astype('int64')

       
def scale_module(df):
    standardScaler = preprocessing.StandardScaler()
    df_standard_scaled = standardScaler.fit_transform(df)
    df_standard_scaled = pd.DataFrame(df_standard_scaled, columns = df.columns)
       

scale_module(df)
print(df.head(10))
y=df['Order_Demand']
X=df.drop(['Order_Demand'],1)
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=0)

#Grouping demand to identify datewise sales
demand_grouped = df.groupby(['Product_Code','Date']).agg(total_sale=('Order_Demand','sum')).reset_index()
#calculating average and standard deviation
cv_data = demand_grouped.groupby('Product_Code').agg(average=('total_sale','mean'),sd=('total_sale','std')).reset_index()
#calculating CV squared
cv_data['cv_sqr']=(cv_data['sd']/cv_data['average'])**2
print(cv_data.head(10))



