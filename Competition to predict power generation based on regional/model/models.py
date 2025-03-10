# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:51:49 2024

@author: StanOWO
"""

# In[] Log Control
import logging
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def startLog(modelName):
    logger = logging.getLogger(__name__)
    logger.info(f"Start running {modelName} model")
    
def finishLog(modelName, val_mse, fileName):
    logger = logging.getLogger(__name__)
    logger.info(f"MSE of Validation Set: {val_mse}")
    logger.info(f"Add file name:{fileName}")
    logger.info(f"Finish running {modelName} model")
    
# In[] Model control
def controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)
    
    val_mse = mean_squared_error(Y_val, Y_pred)
    
    fileName = f"{modelName}_{int(val_mse)}.csv"
    
    return Y_pred, val_mse, fileName, model
    
# In[] Simple regression model
from sklearn.linear_model import LinearRegression

def runSimpleRegression(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)
    
    model = LinearRegression()
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)
    
    return model, Y_pred, fileName, val_mse

# In[] Multiple regression model
import statsmodels.api as sm



def runMultipleRegression(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)
    
    model=sm.OLS(exog=X_train, endog=Y_train).fit()
        
    Y_pred = model.predict(X_val)
    
    val_mse = mean_squared_error(Y_val, Y_pred)
    
    fileName = f"{modelName}_{int(val_mse)}.csv"

    finishLog(modelName, val_mse, fileName)
    
    return model, Y_pred, fileName, val_mse

# In[] SVR model
from sklearn.svm import SVR

def runSVR(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)
    
    model = SVR(kernel='rbf')
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)

    finishLog(modelName, val_mse, fileName)
    
    return model, Y_pred, fileName, val_mse

# In[] Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor

def runRandomForestRegressor(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = RandomForestRegressor(
        n_estimators=100,           # 樹的數量(1~無窮大)
        criterion='squared_error',  # 損失函數(測量分裂品質) ("squared_error" 或 "absolute_error")
        max_depth=None,             # 樹的最大深度(1~無窮大，默認為無限制)
        min_samples_split=2,        # 分裂內部節點所需的最小樣本數(2~無窮大)
        min_samples_leaf=1,         # 葉子節點所需的最小樣本數(1~無窮大)
        min_weight_fraction_leaf=0, # 葉子節點的最小樣本權重(0.0~0.5)
        max_features=None,          # 每次分裂考慮的最大特徵數("auto", "sqrt", "log2" 或 1~特徵數量)
        max_leaf_nodes=None,        # 最大葉子節點數量(2~無窮大，None 表示無限制)
        bootstrap=True,             # 是否使用隨機取樣(True 或 False)
        oob_score=False,            # 是否計算袋外樣本得分(True 或 False)
        n_jobs=-1,                  # 使用的處理器數量(-1 表示使用所有可用處理器)
        random_state=random_key,    # 控制隨機性(整數)
        verbose=0,                  # 訓練過程中的輸出詳情(0 表示無輸出，1 或更高顯示更多詳情)
        warm_start=False,           # 是否重用前次訓練的解(True 或 False)
        max_samples=None            # 每棵樹用於訓練的最大樣本數(1~樣本總數)
    )
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] Decision Tree Regression model
from sklearn.tree import DecisionTreeRegressor

def runDecisionTreeRegressor(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = DecisionTreeRegressor(
        criterion='squared_error',   # 測量分裂品質的標準("squared_error" 或 "absolute_error")
        splitter='best',             # 選擇分裂點的方法("best" 或 "random")
        max_depth=14,                # 樹的最大深度(1~無窮大)
        min_samples_split=115,         # 分裂內部節點所需的最小樣本數(2~無窮大)
        min_samples_leaf=20,          # 葉子節點所需的最小樣本數(1~無窮大)
        max_features=None,           # 用於分裂的特徵數量(None 或 1~特徵數量)
        random_state=random_key      # 控制隨機性(整數)
    )
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] Ridge Regression model
from sklearn.linear_model import Ridge

def runRidgeRegression(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = Ridge(
        alpha=1.0,                  # 正則化強度(0.0~無窮大，0 表示無正則化)
        solver='auto',              # 求解方法("auto", "svd", "cholesky", "saga", "lsqr" 等)
        tol=1e-4,                   # 收斂容差(1e-6~1)
        max_iter=1000,              # 最大迭代次數(1~無窮大)
        random_state=random_key     # 控制隨機性(整數)
    )
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] Lasso Regression model
from sklearn.linear_model import Lasso

def runLassoRegression(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = Lasso(
        alpha=0.1,                  # 正則化強度(0.0~無窮大，越高懲罰越強)
        max_iter=1000,              # 最大迭代次數(1~無窮大)
        tol=1e-4,                   # 收斂容差(1e-6~1)
        selection='random',         # 特徵選擇方法("cyclic" 或 "random")
        random_state=random_key     # 控制隨機性(整數)
    )
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] Gradient Boosting Regression model
from sklearn.ensemble import GradientBoostingRegressor

def runGradientBoostingRegressor(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = GradientBoostingRegressor(
        loss='squared_error',       # 損失函數("squared_error", "absolute_error", "huber", "quantile")
        learning_rate=1.0,          # 每次迭代步長(0.0~1.0)
        n_estimators=120,           # 基模型數量(1~無窮大)
        max_depth=5,                # 基模型的最大深度(1~無窮大)
        min_samples_split=2,        # 分裂內部節點所需的最小樣本數(2~無窮大)
        min_samples_leaf=1,         # 葉子節點所需的最小樣本數(1~無窮大)
        subsample=1.0,              # 用於每次訓練的樣本比例(0.0~1.0)
        random_state=random_key     # 控制隨機性(整數)
    )
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] K-Nearest Neighbors (KNN) Regression model
from sklearn.neighbors import KNeighborsRegressor

def runKNN(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = KNeighborsRegressor(
        n_neighbors=95,             # 鄰居數量(1~無窮大)
        weights='uniform',          # 鄰居加權方式("uniform" 或 "distance")
        algorithm='auto',           # 搜索算法("auto", "ball_tree", "kd_tree", "brute")
        leaf_size=30,               # 樹的葉節點大小(1~無窮大)
        p=2                         # 距離度量方式(1=曼哈頓距離, 2=歐幾里得距離)
    )
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] AdaBoost Regression model
from sklearn.ensemble import AdaBoostRegressor

def runAdaBoostRegressor(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = AdaBoostRegressor(
        n_estimators=10,            # 基學習器數量(1~無窮大)
        learning_rate=0.0001,       # 每次迭代對基學習器的權重(0.0~1.0)
        loss='square',              # 更新權重的方法("linear", "square", "exponential")
        random_state=random_key     # 控制隨機性(整數)
    )
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[]
from xgboost import XGBRegressor

def runXGBoostRegressor(X_train, X_val, Y_train, Y_val, random_key, modelName, parameters):
    if parameters:
        n_estimators = parameters["n_estimators"]
        max_depth = parameters["max_depth"]
        learning_rate = parameters["learning_rate"]
    else:
        n_estimators = 400
        max_depth = 6
        learning_rate = 0.1
        
    startLog(modelName)
    logger = logging.getLogger(__name__)
    logger.info(f"parameters: n_estimators:{n_estimators}, max_depth:{max_depth}, learning_rate:{learning_rate}")

    model = XGBRegressor(
        objective='reg:squarederror',  # 目標函數，使用平方誤差
        n_estimators=n_estimators,              # 樹的數量
        max_depth=max_depth,                   # 樹的最大深度
        learning_rate=learning_rate,             # 學習率
        subsample=0.8,                 # 子採樣比率
        colsample_bytree=0.8,          # 每棵樹的列採樣比率
        random_state=random_key,       # 控制隨機性
        device = "cuda:0"
    )

    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)

    print('訓練集: ',model.score(X_train,Y_train))
    print('測試集: ',model.score(X_val,Y_val))
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] ANN model
from model.pytorch_model import RegressionModel

def runAnn(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = RegressionModel(X_train.shape[1])
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] ANN model
from model.pytorch_model import RegressionModel2

def runAnn2(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = RegressionModel2(X_train.shape[1])
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    print('訓練集: ',r2_score(Y_train, model.predict(X_train)))
    print('測試集: ',r2_score(Y_val,Y_pred))
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] ANN model
from model.pytorch_model import GRURegressionModel

def runGRU(X_train, X_val, Y_train, Y_val, random_key, modelName):
    startLog(modelName)

    model = GRURegressionModel(X_train.shape[1])
    
    Y_pred, val_mse, fileName, model = controlModel(X_train, X_val, Y_train, Y_val, random_key, model, modelName)
    
    print('訓練集: ',r2_score(Y_train, model.predict(X_train)))
    print('測試集: ',r2_score(Y_val,Y_pred))
    finishLog(modelName, val_mse, fileName)

    return model, Y_pred, fileName, val_mse

# In[] Fast run model
def runModel(X_train, X_val, Y_train, Y_val, random_key, simple_modelName, parameters={}):
    simple_modelName = simple_modelName.lower()
    
    if simple_modelName == 'simple':
        model, Y_pred, fileName, val_mse = runSimpleRegression(X_train, X_val, Y_train, Y_val, random_key, "Simple Regression")
    elif simple_modelName == 'multiple':
        model, Y_pred, fileName, val_mse = runMultipleRegression(X_train, X_val, Y_train, Y_val, random_key, "Multiple Regression")
    elif simple_modelName == 'svr':
        model, Y_pred, fileName, val_mse = runSVR(X_train, X_val, Y_train, Y_val, random_key, "SVR")
    elif simple_modelName == 'randomforest':
        model, Y_pred, fileName, val_mse = runRandomForestRegressor(X_train, X_val, Y_train, Y_val, random_key, "Random Forest Regression")
    elif simple_modelName == 'decisiontree':
        model, Y_pred, fileName, val_mse = runDecisionTreeRegressor(X_train, X_val, Y_train, Y_val, random_key, "Decision Tree Regression")
    elif simple_modelName == 'ridge':
        model, Y_pred, fileName, val_mse = runRidgeRegression(X_train, X_val, Y_train, Y_val, random_key, "Ridge Regression")
    elif simple_modelName == 'lasso':
        model, Y_pred, fileName, val_mse = runLassoRegression(X_train, X_val, Y_train, Y_val, random_key, "Lasso Regression")
    elif simple_modelName == 'gradient':
        model, Y_pred, fileName, val_mse = runGradientBoostingRegressor(X_train, X_val, Y_train, Y_val, random_key, "Gradient Boosting Regression")
    elif simple_modelName == 'knn':
        model, Y_pred, fileName, val_mse = runKNN(X_train, X_val, Y_train, Y_val, random_key, "KNN")
    elif simple_modelName == 'ada':
        model, Y_pred, fileName, val_mse = runAdaBoostRegressor(X_train, X_val, Y_train, Y_val, random_key, "Ada Boost Regression")
    elif simple_modelName == 'ann':
        model, Y_pred, fileName, val_mse = runAnn(X_train, X_val, Y_train, Y_val, random_key, "Artificial Neural Network")
    elif simple_modelName == 'ann2':
        model, Y_pred, fileName, val_mse = runAnn2(X_train, X_val, Y_train, Y_val, random_key, "Artificial Neural Network ver2")
    elif simple_modelName == 'gru':
        model, Y_pred, fileName, val_mse = runGRU(X_train, X_val, Y_train, Y_val, random_key, "GRU Network")
    elif simple_modelName == 'xgb' or simple_modelName == 'xgboost':
        model, Y_pred, fileName, val_mse = runXGBoostRegressor(X_train, X_val, Y_train, Y_val, random_key, "XGBoost", parameters)
    else:
        logger = logging.getLogger(__name__)
        logger.info("Can't not find the model")
        return None, 0
    return  model, Y_pred, fileName, val_mse
