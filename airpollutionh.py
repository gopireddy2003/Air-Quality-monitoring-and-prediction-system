# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:55:49 2023

@author: G.HarshaVardhanReddy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
d=pd.read_csv(r"C:\Users\G.HarshaVardhanReddy\Desktop\city_day.csv",parse_dates = ["Date"])
d
d.head()
d.tail()
d.shape
d.columns
d.describe()
d.info()
d.isnull().sum()

pmean=d["PM2.5"].mean()
d["PM2.5"].fillna(pmean,inplace=True)
pmmean=d["PM10"].mean()
d["PM10"].fillna(pmmean,inplace=True)
nmean=d["NO"].mean()
d["NO"].fillna(nmean,inplace=True)
nomean=d["NO2"].mean()
d["NO2"].fillna(nomean,inplace=True)
noxmean=d["NOx"].mean()
d["NOx"].fillna(noxmean,inplace=True)
nhmean=d["NH3"].mean()
d["NH3"].fillna(nhmean,inplace=True)
cmean=d["CO"].mean()
d["CO"].fillna(cmean,inplace=True)
smean=d["SO2"].mean()
d["SO2"].fillna(smean,inplace=True)
omean=d["O3"].mean()
d["O3"].fillna(omean,inplace=True)
bmean=d["Benzene"].mean()
d["Benzene"].fillna(bmean,inplace=True)
tmean=d["Toluene"].mean()
d["Toluene"].fillna(tmean,inplace=True)
xmean=d["Xylene"].mean()
d["Xylene"].fillna(xmean,inplace=True)
amean=d["AQI"].mean()
d["AQI"].fillna(amean,inplace=True)

d.isnull().sum()
d
sns.heatmap(d.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.scatter(d.CO,d.AQI,color='green',marker='*')

plt.figure(figsize=(8,8))
sns.heatmap(d.corr(),annot=True)

x=d.iloc[:,2:13].values
y=d.iloc[:,-2].values
x
y

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(xtrain,ytrain)
ypred=linreg.predict(xtest)
linreg.intercept_
linreg.coef_
Accuracy=linreg.score(xtest,ytest)
print(Accuracy)
 
plt.scatter(ytest,ypred)
sns.distplot((ytest-ypred),bins=50)
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse,r2_score
print(f"MAE:-{mae(ytest,ypred)}")
print(f"MSE:-{mse(ytest,ypred)}")
print(f"RMSE:-{np.sqrt(mse(ytest,ypred))}")
print(f"R-squared:-{r2_score(ytest,ypred)}")
out=linreg.predict([[67.450578,118.127103,0.97,15.69,16.46,23.483476,0.97,24.55,34.06,3.68000,5.500000]])
out

def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x > 50 and x <= 100:
        return "Satisfactory"
    elif x > 100 and x <= 200:
        return "Moderate"
    elif x > 200 and x <= 300:
        return "Poor"
    elif x > 300 and x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return '0'
    
#0-->Good
#1-->Satisfactory
#2-->moderate
#3-->poor
#4-->Very poor
#5-->Severe
    
get_AQI_bucket(out)

from sklearn.ensemble import RandomForestRegressor
# Initialize your random forest regressor
rf = RandomForestRegressor()
from sklearn.model_selection import train_test_split
x=d.iloc[:,2:13].values
y=d.iloc[:,-2].values
xtrain1,xtest1,ytrain1,ytest1= train_test_split(x,y,test_size=0.3,random_state=0)
# Train your model on the training data
rf.fit(xtrain1, ytrain1)
Accuracy=rf.score(xtest1,ytest1)
print(Accuracy)
out1=rf.predict([[67.450578,118.127103,0.97,15.69,16.46,23.483476,0.97,24.55,34.06,3.68000,5.500000]])
out1
get_AQI_bucket(out1)
