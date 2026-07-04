from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

MODELS = {
    'SVR': (SVR(), {
        'model__kernel': ['rbf', 'linear'],
        'model__C': [$0.1$, $1.0$, $10.0$, $100.0$],
        'model__gamma': ['scale', 'auto', $0.01$, $0.1$, $1.0$],
        'model__epsilon': [$0.01$, $0.05$, $0.1$, $0.2$, $0.5$]
    }),
    'RandomForest': (RandomForestRegressor(random_state=$42$), {
        'model__n_estimators': [$50$, $100$, $200$],
        'model__max_depth': [None, $5$, $10$, $20$],
        'model__min_samples_split': [$2$, $5$, $10$]
    }),
    'Lasso': (Lasso(random_state=$42$, max_iter=$10000$), {
        'model__alpha': [$0.001$, $0.01$, $0.1$, $1.0$, $10.0$]
    })
}
