from madlan_data_prep import prepare_data
import pandas as pd
import numpy as np
import pickle

df = pd.read_excel('output_all_students_Train_v10.xlsx')

df = prepare_data(df)

# Select relevant columns and filter the data
relevant_cols = ['City', 'type', 'room_number', 'Area', 'num_of_images',
                 'hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'condition', 'hasAirCondition',
                 'hasBalcony', 'hasMamad', 'handicapFriendly', 'entranceDate', 'furniture',
                 'total_floors', 'floor', 'price']
data = df[relevant_cols]

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)

cat_cols = [col for col in X_train.columns if (X_train[col].dtypes=='O')]
num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O']

numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('scaling', MinMaxScaler())
])
categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False)),
    ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

column_transformer = ColumnTransformer([
     ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)
    ], remainder='drop')

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipeline = make_pipeline(column_transformer, ElasticNet(alpha=0.1, l1_ratio=0.95))
    scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
mean_rmse = rmse_scores.mean()
std_rmse = rmse_scores.std()

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mse =mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE on test set: ", rmse)

# pickle.dump(pipeline, open("trained_model.pkl","wb"))
