# -*- coding: utf-8 -*-
"""
Modified version of file from the deep hazard analysis repo - github.com/ketencimert/deep-hazard-analysis

@author: Mert
"""

import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper

from pycox.datasets import flchain
from pycox.datasets import support
from pycox.datasets import metabric
from pycox.datasets import gbsg
from pycox.datasets import nwtco
import torch

from auton_lab.auton_survival import datasets, preprocessing

def one_hot_encode(dataframe, column):
    categorical = pd.get_dummies(dataframe[column], prefix=column)
    dataframe = dataframe.drop(column, axis=1)
    return pd.concat([dataframe, categorical], axis=1, sort=False)

def load_dataset(
        dataset='SUPPORT',
        scale_time=False
        ):
    if dataset.lower() == 'support':

        features = support.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])
        features_ = dict()

        feature_names = [
            'age',
            'sex',
            'race',
            'number of comorbidities',
            'presence of diabetes',
            'presence of dementia',
            'presence of cancer',
            'mean arterial blood pressure',
            'heart rate',
            'respiration rate',
            'temperature',
            'white blood cell count',
            "serum’s sodium",
            "serum’s creatinine",
             ]

        for i, key in enumerate(feature_names):
            features_[key] = features.iloc[:,i]
        features = pd.DataFrame.from_dict(features_)

        cat_feats = [
            'sex',
            'race',
            'presence of diabetes',
            'presence of dementia',
            'presence of cancer'
            ]
        num_feats = [
            key for key in features.keys() if key not in cat_feats
            ]
        
    elif dataset.lower() == 'metabric':
        features = metabric.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])

        cat_feats = ['x4', 'x5', 'x6', 'x7']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'gbsg':
        features = gbsg.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])

        cat_feats = ['x0', 'x1', 'x2']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'pbc':
        features, t, e = datasets.load_dataset('PBC')
        features = pd.DataFrame(features)
        outcomes = pd.DataFrame([t,e]).T
        outcomes = outcomes.rename(columns={0:'time',1:'event'})

        features = pd.DataFrame(features)
        cat_feats = []
        num_feats = [key for key in features.keys() if key not in cat_feats]
    
    elif dataset.lower() == 'framingham':
        features, t, e = datasets.load_dataset('FRAMINGHAM')
        features = pd.DataFrame(features)
        outcomes = pd.DataFrame([t,e]).T
        outcomes = outcomes.rename(columns={0:'time',1:'event'})

        features = pd.DataFrame(features)
        cat_feats = []
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'nwtco':
        features = nwtco.read_df()
        outcomes = features[['rel', 'edrel']]
        outcomes = outcomes.rename(
            columns={'rel':'event', 'edrel':'time'}
            )
        features = features.drop(columns=['rel', 'edrel'])

        cat_feats = ['stage']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'tcga':

        data = pd.read_csv(
            './TCGA_tables/TCGA_merged.csv'
        )

        outcomes = data[['time', 'event']]
        features = data[[x for x in data.keys() if x not in [*outcomes.keys(), 'PatientID']]]

        num_feats = [key for key in features.keys()]
        cat_feats = []

    elif dataset.lower() == 'nacd':
            
        data = pd.read_csv(
            './data/NACD.csv'
        )

        outcomes = data[['time', 'delta']]
        outcomes = outcomes.rename(
            columns={'delta':'event', 'time':'time'}
            )
        features = data[[x for x in data.keys() if x not in [*outcomes.keys()]]]

        num_feats = [key for key in features.keys()]
        cat_feats = []

    elif dataset.lower() == 'flchain':
        features = flchain.read_df()
        outcomes = features[['death', 'futime']]
        outcomes = outcomes.rename(
            columns={'death':'event', 'futime':'time'}
            )
        features = features.drop(columns=['death', 'futime'])

        cat_feats = ['flc.grp', 'mgus', 'sex']
        num_feats = [key for key in features.keys() if key not in cat_feats]
    
    elif dataset.lower() == 'churn':

        features = pd.read_csv(
            './data/churn.csv'
        )

        outcomes = features[['churned', 'months_active']]
        outcomes = outcomes.rename(
            columns={'churned':'event', 'months_active':'time'}
            )
        features = features.drop(columns=['churned', 'months_active'])

        cat_feats = ['product_travel_expense', 'product_payroll', 'product_accounting', 'company_size', 'us_region']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'employee_attrition':
            
        features = pd.read_csv(
            './data/employee_attrition.csv'
        )

        outcomes = features[['left', 'time_spend_company']]
        outcomes = outcomes.rename(
            columns={'left':'event', 'time_spend_company':'time'}
            )
        features = features.drop(columns=['left', 'time_spend_company'])

        cat_feats = ['department', 'salary']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    features = preprocessing.Preprocessor().fit_transform(
        cat_feats=cat_feats,
        num_feats=num_feats,
        data=features,
        )

    outcomes.time = outcomes.time + 1e-15

    if scale_time:
        outcomes.time = (
            outcomes.time - outcomes.time.min()
            ) / (
                outcomes.time.max() - outcomes.time.min()
                ) + 1e-15

    return outcomes, features

def generate_data(n_samples=10000, setting=1, frac_early_cens=0.15, threshold_early_cens=0.5, frac_early_surv=0.15, threshold_early_surv=0.5, num_features_multi = 40, eot=10):
    assert setting in [1, 2, 3, 4, 5, 6]
    assert 0 <= frac_early_cens <= 1
    assert 0 <= threshold_early_cens <= 1
    assert (n_samples * (1 - threshold_early_cens)) > (n_samples * frac_early_cens)
    if setting in [1, 2]:
        X = np.random.uniform(low=0.0, high=4.0, size=(n_samples, num_features_multi // 10))
    elif setting in [3, 4]:
        X = np.random.uniform(low=0.0, high=4.0, size=(n_samples, num_features_multi // 2))
    else:
        X = np.random.uniform(low=0.0, high=4.0, size=(n_samples, num_features_multi))
    X.sort(axis=0)
    
    T, C, mu_T, mu_C, std_T, std_C = generate_T_C(X, setting, n_samples, frac_early_cens, frac_early_surv, threshold_early_cens, threshold_early_surv, eot=eot)

    data = {
            'duration': np.minimum(T, C),
            'event': (T < C).astype(int)
        }

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    if setting in [4]:
        df = df.assign(duration=data['duration'], event=data['event'], T=T, C=C, mu_T=mu_T, mu_C=mu_C, std_C=std_C, std_T=std_T)
    else:
        df = df.assign(duration=data['duration'], event=data['event'], T=T, C=C, mu_T=mu_T, mu_C=mu_C, std_T=std_T)
        
    return df

def generate_T_C(X, setting, n_samples, frac_early_cens=0.15, frac_early_surv=0.15, threshold_early_cens=0.5, threshold_early_surv=0.5, eot=10):
    if setting==1:
        mu_C = 5
        C = np.random.exponential(scale=mu_C, size=n_samples)

        mu_T = np.sqrt(X[:, 0])
        std_T = 1
        T = np.exp(np.random.normal(loc=mu_T, scale=std_T, size=n_samples))

    elif setting==2:
        mu_C = 10
        C = np.random.exponential(scale=mu_C, size=n_samples)

        mu_T = (2 - np.sqrt(X[:, 3])) * (X[:, 1] > 2) + np.sqrt(X[:, 3]) * (X[:, 0] <= 2)
        std_T = 2
        T = np.exp(np.random.normal(loc=mu_T, scale=std_T, size=n_samples))
    elif setting==3:
        mu_C = 1. / (0.01 + ((4 - X[:, 0])/50.))
        C = np.random.exponential(scale=mu_C, size=n_samples)

        mu_T = (2 - np.sqrt(X[:, 4])) * (X[:, 0] > 2) + X[:, 3] * (X[:, 0] <= 2)
        std_T = 1.5
        T = np.exp(np.random.normal(loc=mu_T, scale=std_T, size=n_samples))
    elif setting==4:
        mu_C = 2 + (2 - X[:, 0])/50
        std_C = 1.5
        C = np.exp(np.random.normal(loc=mu_C, scale=std_C, size=n_samples))

        mu_T = 3 * (X[:, 0] > 2) + 1.5 * X[:, 3] * (X[:, 0] <= 2)
        std_T = 1.5
        T = np.exp(np.random.normal(loc=mu_T, scale=std_T, size=n_samples))
    elif setting==5:
        mu_C = 1. / (X[:, 3] / 40. + X[:, 0] / 40.)
        C = np.random.exponential(scale=mu_C, size=n_samples)

        mu_T = 0.126 * (X[:, 0] + np.minimum(X[:, 2] * X[:, 4], X[:, 1] * X[:, 3])) 
        std_T = 2
        T = np.exp(np.random.normal(loc=mu_T, scale=std_T, size=n_samples))
    elif setting==6:
        mu_C = 1. / (X[:, -1] / 40. + 1. / 20.)
        C = np.random.exponential(scale=mu_C, size=n_samples)

        mu_T = 0.126 * (X[:, 0] + np.sqrt(X[:, 2] * X[:, 4])) + 1
        std_T = (X[:, 1] + 2.) / 2.
        T = np.exp(np.random.normal(loc=mu_T, scale=std_T, size=n_samples))
    random_indecies_cens = np.random.choice(int(n_samples*(threshold_early_cens)), size=int(n_samples*frac_early_cens), replace=False)
    C[random_indecies_cens] = 1e-1
    random_indecies_surv = np.random.choice(int(n_samples*(threshold_early_surv)), size=int(n_samples*frac_early_surv), replace=False)
    T[random_indecies_surv] = np.exp(np.random.normal(loc=-np.log(np.e), scale=2e-1, size=int(n_samples*frac_early_surv)))
    # if setting in [5]:
    #     C = np.minimum(C, 3)
    # else:
    C = np.minimum(C, eot)

    # Define std_C if not defined
    if setting != 4:
        std_C = None

    return T, C, mu_T, mu_C, std_T, std_C

def generate_synthetic(setting, args, frac_early_cens=0.15, threshold_early_cens=0.5, frac_early_surv=0.15, threshold_early_surv=0.5, num_features_multi=40, eot=10):
    df_train = generate_data(n_samples=args['n_samples'], setting=setting, frac_early_cens=frac_early_cens, threshold_early_cens=threshold_early_cens, frac_early_surv=frac_early_surv, threshold_early_surv=threshold_early_surv, eot=eot)
    df_cal = df_train.sample(frac=0.5)
    df_train = df_train.drop(df_cal.index)
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=0.25)
    df_train = df_train.drop(df_val.index)
    df_train = df_train.reset_index()
    # print(f'Generated synthetic data with {df_train.shape[0]} training samples, {df_cal.shape[0]} calibration samples, {df_val.shape[0]} validation samples, and {df_test.shape[0]} test samples')
    
    num_features = num_features_multi if setting in [5,6] else (num_features_multi // 2 if setting in [3,4] else num_features_multi // 10)
    leave = [(col, None) for col in [f'feat_{i}' for i in range(num_features)]]
    x_mapper = DataFrameMapper(leave)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_cal = x_mapper.transform(df_cal).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    in_features = x_train.shape[1]
    return df_train,df_test,df_cal,df_val,x_train,x_val, x_cal, x_test, in_features

def get_real_dataset(dataset, args):
    dataset_dfs = load_dataset(dataset)
    # merge the two dataframes in the tuple dataset_dfs by the index
    df_train = pd.merge(dataset_dfs[0], dataset_dfs[1], left_index=True, right_index=True).astype('float32')
    df_train.rename(columns={'time': 'duration'}, inplace=True)
    df_test = df_train.sample(frac=0.4)
    df_train = df_train.drop(df_test.index)
    df_cal = df_test.sample(frac=0.25)
    df_test = df_test.drop(df_cal.index)
    df_val = df_test.sample(frac=0.5)
    df_test = df_test.drop(df_val.index)
    
    x_train = df_train.drop(columns=['duration', 'event']).values.astype('float32')
    in_features = x_train.shape[1]
    x_cal = df_cal.drop(columns=['duration', 'event']).values.astype('float32')
    x_val = df_val.drop(columns=['duration', 'event']).values.astype('float32')
    x_test = df_test.drop(columns=['duration', 'event']).values.astype('float32')

    return df_train,df_test,df_cal,df_val,x_train,x_val, x_cal, x_test, in_features