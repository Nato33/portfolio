#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
'''sns.set()'''


from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
pd.set_option('max_columns', 200)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Understanding the training set

train.head() # Gives column headers as well as first five rows
train.shape #Obviously gives the shape of the 'matrix'
Quantity_info_train = train.describe() # Gives numerical column data
Categorical_info_train = train.describe(include=['O']) #describe(include = ['O']) will show the descriptive statistics of object data types.
train.info() #We use info() method to see more information of our train dataset.
train_null_count = train.isnull().sum() #sum of all Nan values per column

#Understanding the test Set

test.head() # Gives column headers as well as first five rows
test.shape #Obviously gives the shape of the 'matrix'
Quantity_info_test = test.describe() # Gives numerical column data
Categorical_info_test = test.describe(include=['O']) #describe(include = ['O']) will show the descriptive statistics of object data types.
test.info() #We use info() method to see more information of our train dataset.
test_null_count = test.isnull().sum() #sum of all Nan values per column

#Feature Extraction for  training set

X_train = train.drop(['Alley', 'FireplaceQu', 'PoolQC','Fence','MiscFeature'], axis=1)
X_train['LotFrontage'].fillna(X_train['LotFrontage'].median(), inplace = True)
X_train['MasVnrArea'].fillna(X_train['MasVnrArea'].median(), inplace = True)
X_train['MasVnrType'] = X_train['MasVnrType'].fillna('None')
X_train['BsmtQual'] = X_train['BsmtQual'].fillna('None')
X_train['BsmtCond'] = X_train['BsmtCond'].fillna('None')
X_train['BsmtExposure'] = X_train['BsmtExposure'].fillna('None')
X_train['BsmtFinType1'] = X_train['BsmtFinType1'].fillna('None')
X_train['BsmtFinType2'] = X_train['BsmtFinType2'].fillna('None')
X_train['GarageType'] = X_train['GarageType'].fillna('None')
X_train['GarageFinish'] = X_train['GarageFinish'].fillna('None')
X_train['GarageCond'] = X_train['GarageCond'].fillna('None')
X_train_null_count = X_train.isnull().sum()

X_train['YrBltAndRemod'] = X_train['YearBuilt'] + X_train['YearRemodAdd']
X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']

X_train['Total_sqr_footage'] = (X_train['BsmtFinSF1'] + X_train['BsmtFinSF2'] +
                                 X_train['1stFlrSF'] + X_train['2ndFlrSF'])

X_train['Total_Bathrooms'] = (X_train['FullBath'] + (0.5 * X_train['HalfBath']) +
                               X_train['BsmtFullBath'] + (0.5 * X_train['BsmtHalfBath']))

X_train['Total_porch_sf'] = (X_train['OpenPorchSF'] + X_train['3SsnPorch'] +
                              X_train['EnclosedPorch'] + X_train['ScreenPorch'] +
                             X_train['WoodDeckSF'])
X_train['haspool'] = X_train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
X_train['has2ndfloor'] = X_train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
X_train['hasgarage'] = X_train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
X_train['hasbsmt'] = X_train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
X_train['hasfireplace'] = X_train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


drop_col = ['Exterior2nd', 'GarageYrBlt', 'Condition2', 'RoofMatl', 'Electrical',
            'HouseStyle','Exterior1st', 'Heating','GarageQual','Utilities','SaleType', 'MSZoning', 'FunctioNaNl', 'KitchenQual']
X_train.drop(drop_col, axis = 1,inplace = True)



#Feature Extraction for test set



test = test.drop(['Alley', 'FireplaceQu', 'PoolQC','Fence','MiscFeature'], axis=1)

test['LotFrontage'].fillna(test['LotFrontage'].median(), inplace = True)
test['MasVnrArea'].fillna(test['MasVnrArea'].median(), inplace = True)
test['GarageCars'].fillna(test['GarageCars'].median(), inplace = True)
test['GarageArea'].fillna(test['GarageArea'].median(), inplace = True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].median(), inplace = True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].median(), inplace = True)
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median(), inplace = True)
test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].median(), inplace = True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(), inplace = True)
test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['BsmtQual'] = test['BsmtQual'].fillna('None')
test['BsmtCond'] = test['BsmtCond'].fillna('None')
test['BsmtExposure'] = test['BsmtExposure'].fillna('None')
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('None')
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('None')
test['GarageType'] = test['GarageType'].fillna('None')
test['GarageFinish'] = test['GarageFinish'].fillna('None')
test['GarageCond'] = test['GarageCond'].fillna('None')

test['YrBltAndRemod'] = test['YearBuilt'] + test['YearRemodAdd']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] +
                                 test['1stFlrSF'] + test['2ndFlrSF'])

test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +
                               test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +
                              test['EnclosedPorch'] + test['ScreenPorch'] +
                              test['WoodDeckSF'])
test['haspool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test['has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['hasgarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test['hasbsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['hasfireplace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

drop_col = ['Exterior2nd', 'GarageYrBlt', 'Condition2', 'RoofMatl', 'Electrical',
            'HouseStyle','Exterior1st', 'Heating','GarageQual','Utilities','SaleType', 'MSZoning', 'FunctioNaNl', 'KitchenQual']
test.drop(drop_col, axis = 1,inplace = True)

test = test.fillna(0)
test_null_count = test.isnull().sum()



#(Practically unuasable) Correlation Heatmap
plt.figure(figsize=(30,10))
sns.heatmap(test.corr(),cmap='coolwarm',annot = True)
plt.show()


# Making train and test sets
X_train = X_train.drop(['SalePrice'], axis = 1)
y_train = train['SalePrice']



# One Hot Encoding the categorical data. This is where I am most curious to see the difference in scores 
# When deciding to use One Hot Encoder vs. Pandas' "get dummies".

X_train = pd.get_dummies(X_train, drop_first = True)

#Splitting the data into train and test sets
X_train ,X_test ,y_train ,y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)



#Running Random Forest Regression ensemble model
RFR = RandomForestRegressor(n_estimators = 50)
RFR.fit(X_train,y_train)


#Making our predictions with the model
y_pred = RFR.predict(X_test)


# Calculating the Root Mean Squared Logarithmic Error
print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, y_pred)))


# how the model predicts against the actual known value
plt.figure(figsize=(15,8))
plt.scatter(y_test,y_pred, c='green')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()



test = pd.get_dummies(test, drop_first = True)


test_prediction = RFR.predict(test)


test_pred = pd.DataFrame(test_prediction, columns=['SalePrice'])


from sklearn import ensemble
model = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 4, min_samples_split = 2,
          learning_rate = 0.01)
model.fit(X_train, y_train)


real_pred = model.predict(X_test)

print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, real_pred)))

# how the model predicts against the actual known value
plt.figure(figsize=(15,8))
plt.scatter(y_test,real_pred, c='red')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


test_pred_2 = model.predict(test)


test_pred = pd.DataFrame(test_pred_2, columns=['SalePrice'])

# Gradient Boosting has the lower error compared with Random Forest Regression
submission2 = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_pred_2})
submission2.to_csv('submission2.csv', index=False,header=True)




'''from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in X_train.columns:
    X_train[col] = labelencoder.fit_transform(X_train[col].astype(str))


for col in X_train.columns:
    X_train[col] = labelencoder.fit_transform(X_train[col].astype(str))


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)



sns.barplot(x='SaleCondition', y='SalePrice', data=train)'''



test = test.drop("Id", axis=1).copy()



'''X_train = X_train.dropna(subset = ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageYrBlt','GarageFinish'])'''

