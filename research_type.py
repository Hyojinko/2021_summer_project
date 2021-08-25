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


def scale_module(df, targetName):
    y=df[targetName]
    X=df.drop([targetName],1)
    X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=0)

    # Normalization with 4 Scaling methods
    maxAbsScaler = preprocessing.MaxAbsScaler()
    minmaxScaler = preprocessing.MinMaxScaler()
    robustScaler = preprocessing.RobustScaler()
    standardScaler = preprocessing.StandardScaler()

    df_maxAbs_scaled_train = maxAbsScaler.fit_transform(X_train)
    df_maxAbs_scaled_train = pd.DataFrame(df_maxAbs_scaled_train, columns=X_train.columns)
    df_maxAbs_scaled_test = maxAbsScaler.fit_transform(X_test)
    df_maxAbs_scaled_test = pd.DataFrame(df_maxAbs_scaled_test, columns=X_test.columns)

    df_minMax_scaled_train = minmaxScaler.fit_transform(X_train)
    df_minMax_scaled_train = pd.DataFrame(df_minMax_scaled_train, columns=X_train.columns)
    df_minMax_scaled_test = minmaxScaler.fit_transform(X_test)
    df_minMax_scaled_test = pd.DataFrame(df_minMax_scaled_test, columns=X_test.columns)

    df_robust_scaled_train = robustScaler.fit_transform(X_train)
    df_robust_scaled_train = pd.DataFrame(df_robust_scaled_train, columns=X_train.columns)
    df_robust_scaled_test = robustScaler.fit_transform(X_test)
    df_robust_scaled_test = pd.DataFrame(df_robust_scaled_test, columns=X_test.columns)

    df_standard_scaled_train = standardScaler.fit_transform(X_train)
    df_standard_scaled_train = pd.DataFrame(df_standard_scaled_train, columns=X_train.columns)
    df_standard_scaled_test = standardScaler.fit_transform(X_test)
    df_standard_scaled_test = pd.DataFrame(df_standard_scaled_test, columns=X_test.columns)

    # Alogirthm
    print("\n------------------------- Using maxAbs scaled dataset -------------------------")
    max_score_maxAbs = algorithm_module(df_maxAbs_scaled_train, df_maxAbs_scaled_test, y_train, y_test)
    print("\n------------------------- Using minMax scaled dataset -------------------------")
    max_score_minMax = algorithm_module(df_minMax_scaled_train, df_minMax_scaled_test, y_train, y_test)
    print("\n------------------------- Using robust scaled dataset -------------------------")
    max_score_robust = algorithm_module(df_robust_scaled_train, df_robust_scaled_test, y_train, y_test)
    print("\n------------------------- Using standard scaled dataset -------------------------")
    max_score_standard = algorithm_module(df_standard_scaled_train, df_standard_scaled_test, y_train, y_test)
