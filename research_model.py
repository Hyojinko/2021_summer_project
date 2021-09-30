import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Historical Product Demand.csv')
df = pd.read_csv('Intermittent.csv', names=['Product_Code','ADI','average','sd','cv_sqr',"category"], header=None)
df=df.drop(["category"],axis=1)
print(df.head(20))
print(df.info())


def oneHotEncode_category(arr,str):
    enc=preprocessing.OneHotEncoder()
    encodedData=enc.fit_transform(arr[[str]])
    encodedDataRecovery=np.argmax(encodedData,axis=1).reshape(-1,1)
    arr[str] = encodedDataRecovery


oneHotEncode_category(df,'Product_Code')



def callParameters(num):
    adaBoostParameters = {
        "loss": ["linear","square","exponential"],
        "random_state": [3,5,10],
        "n_estimators": [3, 5, 10],
        "learning_rate": [0.5,1.0,1.5]
    }
    baggingParameters = {
        "n_estimators": [3, 5, 10],
        "bootstrap":[True,False],
        "warm_start": [True,False],
        "random_state": [3,5,10]
    }
    gradientBoostParameters={
        "loss": ["ls","lad"],
        "random_state": [3,5,10],
        "n_estimators": [3, 5, 10],
        "learning_rate": [0.5,1.0,1.5]
    }
    randomForestParameters = {
        "min_impurity_decrease": [0.0,1.0,1.5],
        "random_state": [3, 5, 10],
        "n_estimators": [3, 5, 10],
        "max_features": ['auto','sqrt',"log2"]
    }
    extraTreesParameters = {
        "min_impurity_decrease": [0.0, 1.0, 1.5],
        "random_state": [3, 5, 10],
        "n_estimators": [3, 5, 10],
        "max_features": ['auto', 'sqrt', "log2"]
    }
    if num==0:
        return adaBoostParameters
    elif num==1:
        return baggingParameters
    elif num==2:
        return gradientBoostParameters
    elif num==3:
        return randomForestParameters
    else: return extraTreesParameters


def findBestModel (dataset):
    bestscore=0
    i=0
    y = dataset["average"]
    x = dataset.drop(["average"], axis=1)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=0)
    best = []
    for model in [AdaBoostRegressor(), BaggingRegressor(),GradientBoostingRegressor(), RandomForestRegressor(), ExtraTreesRegressor()]:
        tunedModel = GridSearchCV(model, callParameters(i), scoring='neg_mean_squared_error', cv=5)
        tunedModel.fit(train_x, train_y)
        print(i)
        print(tunedModel.best_params_)
        print(tunedModel.best_score_)
        i = i + 1
        if bestscore > tunedModel.best_score_:
            bestscore = tunedModel.best_score_
            bestparams = tunedModel.best_params_
        best.append(bestparams)
        best.append(bestscore)

    return best

print(findBestModel(df))




