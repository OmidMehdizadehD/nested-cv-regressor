from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

MODELS = {
    'SVM': (SVR(), {
        'model__kernel': ['linear', 'rbf'],
        'model__C': [0.1, 1.0, 10.0, 100.0],
        'model__gamma': ['scale', 'auto', 0.01, 0.1],
        'model__epsilon': [0.01, 0.05, 0.1, 0.2] # Assuming 0.05 and 0.1 are the intermediate steps up to 0.2
    }),
    
    'Decision Tree': (DecisionTreeRegressor(random_state=42), {
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }),
    
    'KNN': (KNeighborsRegressor(), {
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2] # 1 for Manhattan, 2 for Euclidean
    }),
    
    'ANN': (MLPRegressor(early_stopping=True, max_iter=100, random_state=42), {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01], # L2 penalty parameter
        'model__learning_rate_init': [0.001, 0.01]
    })
}
