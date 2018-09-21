import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

#---Data Wrangling---
#Load Data
train = pd.read_csv(r"C:/Users/Noxx.inc/Documents/Data science/Kaggle/House Prices/train.csv")
test = pd.read_csv(r"C:/Users/Noxx.inc/Documents/Data science/Kaggle/House Prices/test.csv")
#Data Explore and Engineer
print("skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
target = np.log(train.SalePrice) # Normalizing the Distribution
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
  #Numeric Features
numeric_features = train.select_dtypes(include=[np.number])
  #Correlation between numeric_features
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
  #As OverallQual is highly dependent lets go a bit deeper into it
train.OverallQual.unique() #The OverallQual data are integer values in the interval 1 to 10 inclusive.
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(quality_pivot)
  #Visualizing the OverallQual
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
  #Other Visualizations
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
  #Removing Outliers from GarageArea
train = train[train['GarageArea'] < 1200] # As there are Outliers correspondingly after 1200

  #After removing Outliers from GarageArea
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
  #Handling Null Values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print('\n',nulls) #Check the features with max nulls

  #Non-Numeric features.
categoricals = train.select_dtypes(exclude=[np.number])
print('\n',categoricals.describe())
  #One-Hot Encoding
#print (train.Street.value_counts(), "\n") Gives the categoricals
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True) # As train and test need to be same
  #Visualizing SaleCondition similar to OverallQual
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
def sc_encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(sc_encode)
test['enc_condition'] = test.SaleCondition.apply(sc_encode)
  #SaleCondition after encodeing
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
  #Interpolation
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
  #ReCheck nulls
print(data.isnull().sum().sum())

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.1)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))


#Visualizing Results
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


#Kaggle submission
submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions
print(submission.head())
submission.to_csv('submission1.csv', index=False)
