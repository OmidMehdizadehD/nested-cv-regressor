from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer, 
    MinMaxScaler, MaxAbsScaler, QuantileTransformer
)
from sklearn.svm import OneClassSVM
import numpy as np

def get_scaler(method='standard')
    method = method.lower()
    scalers = {
        'standard' StandardScaler(),
        'robust' RobustScaler(),
        'minmax' MinMaxScaler(),
        'maxabs' MaxAbsScaler(),
        'power' PowerTransformer(method='yeo-johnson'),
        'quantile' QuantileTransformer(output_distribution='normal')
    }
    if method not in scalers
        raise ValueError(fInvalid scaling method {method})
    return scalers[method]

def osvm_resampler(X, y, contamination=$0.01$)
    
    Fits OneClassSVM and returns inliers. 
    Designed to be used with imblearn's FunctionSampler to prevent data leakage.
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    osvm = OneClassSVM(kernel='rbf', contamination=contamination)
    preds = osvm.fit_predict(X_scaled)
    
    inlier_mask = preds == $1$
    return X[inlier_mask], y[inlier_mask]
