import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('adult.csv', na_values = '#NAME?')

df['income'] = [ 0 if x == '<=50K' else 1 for x in df['income']]

X = df.drop('income', 1).drop('fnlwgt', 1)
y = df.income

# print(X.head(5))
# print(y.head(5))

# print(X['education']. head(5))

# # print(pd.get_dummies(X['education']).head(5))

# for column_name in X.columns:
# 	if X[column_name].dtypes == 'object':
# 		unique_categories = len(X[column_name].unique())
# 		print(column_name, unique_categories)

# print(X['native_country'].value_counts().sort_values(ascending=False).head(10))

X['native_country'] = ['United-States' if x == 'United-States' else 'Other' for x in df['native_country']]

dummy_list = ['workclass', 'education','marital_status','occupation','relationship','race','sex','native_country']

def dummy(df, dummy_list):
	for x in dummy_list:
		dummies = pd.get_dummies(df[x], prefix=x, dummy_na = False)
		df = df.drop(x, 1)
		df = pd.concat([df, dummies], axis = 1)
	return df

X = dummy(X, dummy_list)
# print(X.head(5))

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer.fit(X)
X = pd.DataFrame(data = imputer.transform(X), columns = X.columns)

# print(X.isnull().sum().sort_values(ascending= False). head())

def find_outliers_tukey(x):
	q1 = np.percentile(x, 25)
	q3 = np.percentile(x, 75)
	_ = q3 - q1
	tukey_formula_floor = q1 - 1.5 * _
	tukey_formula_ceiling = q3 + 1.5 * _
	outlier_indices = list(x.index[(x< tukey_formula_floor)| (x > tukey_formula_ceiling)])
	outlier_values = list(x[outlier_indices])

	return outlier_indices, outlier_values

tukey_indices, tukey_values = find_outliers_tukey(X['age'])

print (tukey_indices)

from sklearn.decomposition import PCA

pca = PCA(n_components = 10)
X_pca = pd.DataFrame(pca.fit_transform(X))

# print(X_pca.head(5))

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, random_state = 1)

# print(y.shape)
# print(df.shape)

import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k = 20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices = True)
columnnames_selected = [X.columns[x] for x in indices_selected]

X_train_selected = X_train[columnnames_selected]
X_test_selected = X_test[columnnames_selected]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def find_model_performance(X_train, y_train, X_test, y_test):
	model = LogisticRegression()
	model.fit(X_train, y_train)
	y_hat = [x[1] for x in model.predict_proba(X_test)]
	auc = roc_auc_score(y_test, y_hat)
	
	return auc

auc_processed = find_model_performance(X_train_selected, y_train, X_test_selected, y_test)
print(auc_processed)


df_unprocessed = df
df_unprocessed = df_unprocessed.dropna(axis = 0, how = 'any')

for x in df_unprocessed.columns:
	if df_unprocessed[x].dtypes not in ['int32','int64', 'float32', 'float64']:
		df_unprocessed = df_unprocessed.drop(x, 1)

X_unprocessed = df_unprocessed.drop('income', 1).drop('fnlwgt',1)
y_unprocessed = df_unprocessed.income

X_train_unprocessed, X_test_unprocessed, y_train_unprocessed, y_test_unprocessed = train_test_split(X_unprocessed, y_unprocessed, test_size = 0.7, random_state = 1)

auc_unprocessed = find_model_performance(X_train_unprocessed, y_train_unprocessed, X_test_unprocessed, y_test_unprocessed)

print(auc_unprocessed)


























