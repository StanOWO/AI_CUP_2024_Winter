# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:46:46 2024

@author: StanOWO
"""
import pandas as pd
import logging
from utils.preprocessor import preprocessing, split_test_data
from model.models import runModel
import sys
from tqdm import tqdm

def isMean_test_prediction(X_pred, test_dataset, model):
    # 預測未來 10 分鐘的平均值
    X_pred_avg = []
    for i, row in tqdm(X_pred.iterrows(), total=X_pred.shape[0], desc="columns"):  # 對測試集的每一行進行處理
        h = row['h']
        m = row['min']
        
        ten_minute_predictions = []
        for offset in range(10):  # 預測未來 10 分鐘
            new_row = row.copy()
            new_row['h'] = h
            new_row['min'] = m + offset
            
            # 預測新的特徵行
            prediction = model.predict(new_row.to_frame().T)
            ten_minute_predictions.append(prediction[0])  # 添加預測結果
        
        # 計算十個預測的平均值
        isMean_prediction = sum(ten_minute_predictions) / len(ten_minute_predictions)
        row['prediction'] = isMean_prediction
        X_pred_avg.append(row)
    
    return isMean_prediction

def prediction_test_set(model, X_pred, sequence, test_dataset, result_dir, isMean):
    if not model:
        sys.exit(0)
        
    logger.info("Starting prediction on test set")
    
    if isMean:
        Y_test = pd.DataFrame(isMean_test_prediction(X_pred, test_dataset, model), columns=["答案"])
    else:
        Y_test = pd.DataFrame(model.predict(X_pred), columns=["答案"])
    Y_test = Y_test.round(2)
    
    result = pd.concat([sequence, Y_test], axis=1)    
    
    logger.info(f"Saving new results to {result_dir}")
    result.to_csv(result_dir, index=False)
    
def multiple_prediction(test_dataset, X_train, X_val, Y_train, Y_val, model_names, basic_columns, feature_columns, random_key):
    logger.info("Starting multiple model prediction")
    logger.info(f"Predict columns：{feature_columns}")
    X_pred = pd.DataFrame()
    for counter, feature_column in enumerate(feature_columns):
        logger.info(f"Starting predict column：{feature_column}")

        model, _, _ = runModel(X_train, X_val, Y_train[feature_column], Y_val[feature_column], model_names[counter], random_key)
        logger.info(f"Finishing predict column：{feature_column}")
        
        Y_pred = model.predict(test_dataset[basic_columns])
        
        X_pred[feature_column] = Y_pred
        
    return X_pred


# In[] indirect prediction with 3 layer
def indirect4_prediction(train_dirs, test_dataset, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, sequence, model_names):
    logger.info("Starting indirect prediction with 4 layers")
    feature_columns_1 = ['WindSpeed(m/s)', 'Pressure(hpa)']
    feature_columns_2 = ['Temperature(°C)', 'Humidity(%)']
    feature_columns_3 = ['Sunlight(Lux)']
    
    random_key = 48169 
    logger.info("Starting preprocessing in layer 1")
    feature_colum = basic_columns 
    X_train, X_val, Y_train, Y_val = preprocessing(train_dirs, feature_colum, feature_columns_1, val_set_size, random_key)
    logger.info("Starting model training in layer 1")
    parameters = { "n_estimators" : 1600 , "max_depth" : 12 , "learning_rate" : 0.08}
    random_key = 48169 
    model, Y_pred, filename, val_mse = runModel(X_train, X_val, Y_train, Y_val, random_key, model_names[0])
    result_dir = result_dir + filename
    logger.info("Starting prediction on test set in layer 1")
    X_pred = test_dataset[feature_colum]
    Y_test = pd.DataFrame(model.predict(X_pred), columns=feature_columns_1)
    test_dataset = pd.concat([test_dataset, Y_test], axis=1)
    
    random_key = 48763
    logger.info("Starting preprocessing in layer 2")
    feature_colum = basic_columns + feature_columns_1
    X_train, X_val, Y_train, Y_val = preprocessing(train_dirs, feature_colum, feature_columns_2, val_set_size, random_key)
    logger.info("Starting model training in layer 2")
    parameters = { "n_estimators" : 600 , "max_depth" : 15 , "learning_rate" : 0.2}
    random_key = 0
    model, Y_pred, filename, val_mse = runModel(X_train, X_val, Y_train, Y_val, random_key, model_names[1], parameters)
    result_dir = result_dir[:-4] + "_" + filename
    logger.info("Starting prediction on test set in layer 2")
    X_pred = test_dataset[feature_colum]
    Y_test = pd.DataFrame(model.predict(X_pred), columns=feature_columns_2)
    test_dataset = pd.concat([test_dataset, Y_test], axis=1)
    
    logger.info("Starting preprocessing in layer 3")
    random_key = 48763
    feature_colum = basic_columns + feature_columns_1 + feature_columns_2
    X_train, X_val, Y_train, Y_val = preprocessing(train_dirs, feature_colum, feature_columns_3, val_set_size, random_key)
    logger.info("Starting model training in layer 3")
    parameters = { "n_estimators" : 1800 , "max_depth" : 14 , "learning_rate" : 0.05}
    random_key = 48763
    model, Y_pred, filename, val_mse = runModel(X_train, X_val, Y_train, Y_val, random_key, model_names[2], parameters)
    result_dir = result_dir[:-4] + "_" + filename
    logger.info("Starting prediction on test set in layer 3")
    X_pred = test_dataset[feature_colum]
    Y_test = pd.DataFrame(model.predict(X_pred), columns=feature_columns_3)
    test_dataset = pd.concat([test_dataset, Y_test], axis=1)
    
    logger.info("Starting preprocessing in layer 4")
    random_key = 48763
    feature_colum = basic_columns + feature_columns_1 + feature_columns_2 + feature_columns_3
    X_train, X_val, Y_train, Y_val = preprocessing(train_dirs, feature_colum, target_columns, val_set_size, random_key)
    logger.info("Starting model training in layer 4")
    parameters = { "n_estimators" : 1800 , "max_depth" : 14 , "learning_rate" : 0.02}
    random_key = 48763
    model, Y_pred, filename, val_mse = runModel(X_train, X_val, Y_train, Y_val, random_key, model_names[3], parameters)
    result_dir = result_dir[:-4] + "_" + filename
    logger.info("Starting prediction on test set in layer 4")
    X_pred = test_dataset[feature_colum]
    Y_test = pd.DataFrame(model.predict(X_pred), columns=["答案"])
    Y_test = Y_test.round(2)
    test_dataset = pd.concat([test_dataset, Y_test], axis=1)
    
    if isMean:
        combine_test_dataset = test_dataset.groupby(['序號','y', 'mo', 'd', 'h', 'LocationCode'], as_index=False).mean()
        combine_test_dataset['序號'] = combine_test_dataset['序號'].astype('longlong')
        combine_test_dataset['序號'] = pd.Categorical(combine_test_dataset['序號'], categories=sequence, ordered=True)
        combine_test_dataset = combine_test_dataset.sort_values(by='序號', ascending=True).reset_index(drop=True)
        result = pd.concat([combine_test_dataset["序號"], combine_test_dataset["答案"]], axis=1)
    else:
        result = pd.concat([test_dataset["序號"], test_dataset["答案"]], axis=1)
    
    logger.info(f"Saving new results to {result_dir}")
    result.to_csv(result_dir, index=False)  


# In[] indirect prediction
def indirect_prediction(train_dirs, test_dataset, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, sequence, model_names, isMean):
    logger.info("Starting indirect prediction")
    
    logger.info("Starting preprocessing")
    X_train, X_val, Y_train, Y_val = preprocessing(train_dirs, basic_columns, feature_columns, val_set_size, random_key)
    
    X_pred = multiple_prediction(test_dataset, X_train, X_val, Y_train, Y_val, model_names, basic_columns, feature_columns, random_key)
    # X_pred = pd.concat([X_pred, test_dataset[basic_columns]], axis=1)
    
    _, Y_pred, filename = runModel(X_train, X_val, Y_train, Y_val, model_names[5], random_key)
    
    model = direct_prediction(train_dirs, test_dataset, result_dir, feature_columns, target_columns, random_key, val_set_size, sequence, model_names, False, isMean)
    
    prediction_test_set(model, X_pred, sequence, test_dataset, result_dir + filename, isMean)

# In[] direct prediction
def direct_prediction(train_dirs, test_dataset, result_dir, feature_columns, target_columns, random_key, val_set_size, sequence, model_names, is_test, isMean):
    logger.info("Starting direct prediction")
    
    logger.info("Starting preprocessing")
    X_train, X_val, Y_train, Y_val = preprocessing(train_dirs, feature_columns, target_columns, val_set_size, random_key)
    
        
    logger.info("Starting model training")
    
    model, Y_pred, filename = runModel(X_train, X_val, Y_train, Y_val, model_names[5], random_key)
    
    
    if is_test:
        X_pred = test_dataset[feature_columns]
        prediction_test_set(model, X_pred, sequence, test_dataset, result_dir + filename, isMean)
        
    return model
    

def main(train_dirs, test_dir, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, model_names, predict_method, isMean):
    logger.info("Starting split testing set")
    test_dataset = pd.read_csv(test_dir)
    test_dataset= test_dataset.drop(["答案"],axis=1)
    test_dataset, sequence = split_test_data(test_dataset)
    
    mean_name = "noAverage"
    if isMean: 
        mean_name = "Average"
        
    if result_dir == "auto":
        result_dir = f"submit/{predict_method}_{mean_name}_"
    
    if predict_method == "direct":
        direct_prediction(train_dirs, test_dataset, result_dir, basic_columns, target_columns, random_key, val_set_size, sequence, model_names, True, isMean)
    elif predict_method == "indirect":
        indirect_prediction(train_dirs, test_dataset, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, sequence, model_names, isMean)
    elif predict_method == "indirect4":
        indirect4_prediction(train_dirs, test_dataset, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, sequence, model_names)
        
# In[]
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filemode='w')
    logger = logging.getLogger(__name__)

    train_dirs = "dataset"
    test_dir = "submit/test_data.csv"
    result_dir = "auto"
    
    basic_columns = ['y', 'mo', 'd', 'h', 'min', 'LocationCode']
    feature_columns = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
    target_columns = ['Power(mW)']
    random_key = 48169

    val_set_size = 0.01
    
    predict_method = "indirect4"
    isMean = True

    fast_model_options = ["simple", "multiple", "ridge", "lasso", "knn", "decisiontree"]
    mediumSpeed_model_options = ["ada", "gradient", "randomforest"]
    slow_model_options = ["svr"]
    
# In[]
    model_names = ["randomforest", "xgb", "xgb", "xgb", "simple", "knn"]
    main(train_dirs, test_dir, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, model_names, predict_method, isMean)
    
# In[]
    # for option in fast_model_options:
    #     model_names = ["simple", "simple", "simple", "simple", "simple", option]
    #     main(train_dirs, test_dir, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, model_names, predict_method, isMean)

# In[]
    # for option in mediumSpeed_model_options:
    #     model_names = ["simple", "simple", "simple", "simple", "simple", option]
    #     main(train_dirs, test_dir, result_dir, basic_columns, feature_columns, target_columns, random_key, val_set_size, model_names, predict_method, isMean)
