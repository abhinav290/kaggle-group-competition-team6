import pandas as pd
import seaborn as sns
import math
import pickle
import numpy as np
from category_encoders import *
import matplotlib.pyplot as plt
from pyod.models.feature_bagging import FeatureBagging
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set()

train_file = "data/tcd-ml-1920-group-income-train.csv"
test_file = "data/tcd-ml-1920-group-income-test.csv"
submission_file = "data/tcd-ml-1920-group-income-submission.csv"
model_file = "model.sav"


def pre_process(input_file):
    df = pd.read_csv(input_file)
    df.columns = ['instance', 'record_year', 'housing', 'crime', 'exp', 'satisfaction', 'gender', 'age', 'country',
                  'city_size', 'profession', 'degree', 'glasses', 'hair_color', 'height', 'extra_income', 'income']



    # Initial Pre-processing
    # Dropping unwanted columns
    df = df.drop(['instance', 'hair_color', 'glasses', 'height'], axis=1)

    # Processing for column record_year
    # Taking median of the data to replace null values
    median_year = math.floor(df.record_year.median())
    df.record_year = df.record_year.fillna(median_year)

    # Processing for column age
    # Taking median of the data to replace null values
    median_age = math.floor(df.age.median())
    df.age = df.age.fillna(median_age)

    # Processing for column gender
    df.gender.replace([0, np.nan,'female','male'], ['unknown', 'unknown','f','m'], inplace=True)
    # df_gender = pd.get_dummies(df.gender)
    # df = pd.concat([df, df_gender], axis=1)
    # df = df.drop('gender', axis=1)


    ## Processing for column profession
    df.profession.fillna('Unknown')
    df.profession = df.profession.str[:3]


    ## Processing for column degree
    df.degree.replace(['no', np.nan], ['No','Unknown'], inplace=True)
    df.degree = df.degree.str[:5]

    ## Processing for column housing
    df.housing.replace([0, 'nA'], 'unknown', inplace=True)

    ## Processing for column experience
    df.exp.replace(['#NUM!'], np.NAN, inplace=True)
    df.exp = pd.to_numeric(df.exp)
    df.exp.fillna(df.exp.median(), inplace=True)

    ## Processing for column satisfaction
    df.satisfaction = df.satisfaction.fillna('unknown')

    ## Processing for column extra_income
    df.extra_income = df.extra_income.str[:-4]
    df.extra_income = pd.to_numeric(df.extra_income)

    columns = ['gender', 'country', 'profession', 'degree']
    for col in columns:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip()

    # Removing predicted outcome variable
    df_income = df.income
    df = df.drop('income', axis=1)

    return df, df_income

def remove_outliers(df, df_income):
    lab_enc = preprocessing.LabelEncoder()
    clf = FeatureBagging(n_jobs=10,verbose=-1)
    clf.fit(df, lab_enc.fit_transform(df_income))
    df = clf.predict(df)



def train_model():
    ## Train  model

    df, df_income = pre_process(train_file)
    df_pred_data, df_pred_income = pre_process(test_file)



    # Target encoding for Country and Profession
    cat_vars = ['country', 'profession', 'satisfaction', 'degree', 'gender']
    enc = TargetEncoder(cat_vars).fit(df, df_income)
    df = enc.transform(df)
    df_pred_data = enc.transform(df_pred_data)

    # Scaling the data
    scaler = preprocessing.StandardScaler().fit(df)
    df_scaled_data = scaler.transform(df)
    df_pred_scaled_data = scaler.transform(df_pred_data)



    # Splitting the data for testing and training
    X_train = df_scaled_data
    Y_train = df_income

    x_train, x_val, y_train, y_val = model_selection.train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)
    trn_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)



    # Training the model.

    params = {
        'max_depth': 20,
        'learning_rate': 0.001,
        "boosting": "gbdt",
        "bagging_seed": 11,
        "metric": 'mse',
        "max_bin": 1024,
        "num_leaves": 200,
        "n_jobs": 10,
        "verbosity": -1,
    }

    clf = lgb.train(params, trn_data, 120000, valid_sets=[trn_data, val_data], verbose_eval=1000, early_stopping_rounds=200)

    # Predicting the values.
    df_pred_income = clf.predict(df_pred_scaled_data)
    pre_val_lgb = clf.predict(x_val)

    # Measuring error.
    val_mse = metrics.mean_squared_error(y_val, pre_val_lgb)
    val_rmse = np.sqrt(val_mse)
    print(val_rmse)

    return df_pred_income

def store_model(model, model_file):
    pickle.dump(model, open(model_file, 'wb'))


# Store output
def store_output(out_pred):
    df_sub = pd.read_csv(submission_file, index_col=False)
    df_sub['Total Yearly Income [EUR]'] = out_pred
    with open(submission_file, 'w') as f:
        df_sub.to_csv(f, index=False, line_terminator='\n')


def find_pattern():
    df, df_income = pre_process(train_file)

    for col in df.columns:
        print(col)
        print(df[col].isna().sum())
        print(df[col].unique())

    #
    # # Show graph plots
    # plt.figure(figsize=(100, 50))
    # plt.xlabel("Country")
    # plt.ylabel("Income in euros")
    # plt.scatter(df.hair_color, df.income, s=1)
    # plt.show()




out_pred = train_model()
store_output(out_pred)

