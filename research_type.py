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

ordinalEncode_category(df, "Warehouse")
ordinalEncode_category(df, "Product_Category")

# Since the "()" has been removed , Now i Will change the data type.
df['Order_Demand'] = df['Order_Demand'].str.replace('(', "", regex=True)
df['Order_Demand'] = df['Order_Demand'].str.replace(')', "", regex=True)
df['Order_Demand'] = df['Order_Demand'].astype('int64')



def scale_module(df):
    standardScaler = preprocessing.StandardScaler()
    df_standard_scaled = standardScaler.fit_transform(df)
    df_standard_scaled = pd.DataFrame(df_standard_scaled, columns=df.columns)


print(df.head(10))

y = df['Order_Demand']
X = df.drop(['Order_Demand'], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Grouping demand to identify datewise sales
demand_grouped = df.groupby(['Product_Code','Date']).agg(total_sale=('Order_Demand','sum')).reset_index()
#calculating average and standard deviation
cv_data = demand_grouped.groupby('Product_Code').agg(average=('total_sale','mean'),sd=('total_sale','std')).reset_index()
#calculating CV squared
cv_data['cv_sqr']=(cv_data['sd']/cv_data['average'])**2
print(cv_data.head(10))

df['Date'] = df['Date'].astype('datetime64[ns]')

prod_by_date=df.groupby(['Product_Code','Date']).agg(count=('Product_Code','count')).reset_index()
skus=prod_by_date.Product_Code.value_counts()
print("SKUS")
print(skus)
new_df= pd.DataFrame()

for i in range(len(skus.index)):
    a= prod_by_date[prod_by_date['Product_Code']==skus.index[i]]
    a['previous_date']=a['Date'].shift(1)
    new_df=pd.concat([new_df,a],axis=0)

print(new_df.info())
new_df['duration']=new_df['Date']- new_df['previous_date']
new_df['Duration']=new_df['duration'].astype(str).str.replace('days','')
new_df['Duration']=pd.to_numeric(new_df['Duration'],errors='coerce')
ADI = new_df.groupby('Product_Code').agg(ADI = ('Duration','mean')).reset_index()
print(ADI)

adi_cv=pd.merge(ADI,cv_data)
print(adi_cv.head(10))

#defining a function for categorization
def category(df):
    a=0

    if((df['ADI']<=1.34) & (df['cv_sqr']<=0.49)):
        a = 'Smooth'
    if((df['ADI']>=1.34) & (df['cv_sqr']>=0.49)):
        a = 'Lumpy'
    if((df['ADI']<1.34) & (df['cv_sqr']>0.49)):
        a = 'Erratic'
    if((df['ADI']>1.34) & (df['cv_sqr']<0.49)):
        a = 'Intermittent'
    return a

#categorizing products based on their forcastability
adi_cv['category']=adi_cv.apply(category,axis=1)
print(adi_cv.head(20))
#categorized list
print(adi_cv['category'].head())


sns.scatterplot(x='cv_sqr',y='ADI',hue='category',data=adi_cv)
plt.show()


