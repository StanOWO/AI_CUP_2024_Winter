# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:46:46 2024

@author: StanOWO
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def decomposition(dataset, x_columns, y_columns=None):
    X = dataset[x_columns]
    Y = dataset[y_columns] if y_columns else None
    return (X, Y) if Y is not None else X

def split_test_data(dataset, isMean=True):
    dataset['y'] = dataset['序號'].astype(str).str[0:4].astype(int).squeeze()
    dataset['mo'] = dataset['序號'].astype(str).str[4:6].astype(int).squeeze()
    dataset['d'] = dataset['序號'].astype(str).str[6:8].astype(int).squeeze()
    dataset['h'] = dataset['序號'].astype(str).str[8:10].astype(int).squeeze()
    dataset['min'] = dataset['序號'].astype(str).str[10:12].astype(int).squeeze()
    dataset['LocationCode'] = dataset['序號'].astype(str).str[12:14].astype(int).squeeze()
    
    sequence = dataset["序號"]
    
    if isMean:
        datasets = []
        for i in range(10):
            dataset_cp = dataset.copy()
            dataset_cp['min'] = dataset['min'] + i
            datasets.append(dataset_cp)
        dataset = pd.concat(datasets, ignore_index=True)
    return dataset, sequence

def split_date_time(dataset) ->  pd.DataFrame:
    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
    
    dataset['y'] = dataset['DateTime'].dt.year.squeeze()
    dataset['mo'] = dataset['DateTime'].dt.month.squeeze()
    dataset['d'] = dataset['DateTime'].dt.day.squeeze()
    dataset['h'] = dataset['DateTime'].dt.hour.squeeze()
    dataset['min'] = dataset['DateTime'].dt.minute.squeeze()
    
    dataset = dataset.drop(columns=['DateTime'])
    
    return  dataset


def concat_dataset(directory:str) -> pd.DataFrame:
    datasets = pd.DataFrame()
    for counter, file_name in enumerate(os.listdir(directory)):
        dataset = pd.read_csv(os.path.join(directory,file_name) )
        datasets = pd.concat([datasets,dataset],ignore_index=True)
    return datasets

def preprocessing(train_dirs, X_cols, Y_cols, test_set_size=0.2, random_key=0):
    dataset = concat_dataset(train_dirs)
    dataset = split_date_time(dataset)
    X, Y = decomposition(dataset, x_columns=X_cols, y_columns=Y_cols)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_set_size, random_state=random_key)
    
    return X_train, X_val, Y_train, Y_val

