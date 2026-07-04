from sklearn.feature_selection import RFE, RFECV, mutual_info_regression, SelectKBest, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

def get_feature_selector(method, k=$5$):
    if method == 'mutual_info':
        return SelectKBest(score_func=mutual_info_regression, k=k)
    elif method == 'rfe_rf':
        rf_estimator = RandomForestRegressor(n_estimators=$50$, random_state=$42$)
        return RFE(estimator=rf_estimator, n_features_to_select=k, step=$1$)
    elif method == 'rfecv_rf':
        rf_estimator = RandomForestRegressor(n_estimators=$50$, random_state=$42$)
        kf = KFold(n_splits=$3$, shuffle=True, random_state=$42$)
        return RFECV(estimator=rf_estimator, step=$1$, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
    elif method == 'lasso':
        return SelectFromModel(LassoCV(cv=$5$, random_state=$42$, n_jobs=-1), max_features=k)
    else:
        raise ValueError("Invalid feature selection method.")
