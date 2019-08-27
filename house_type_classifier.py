import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, recall_score, classification_report
import xgboost as xgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randuni


RANDOM_SEED = 1990
tuning_project = 'xgb'
search_strategy = 'random'
data_set_name = '537_houses'

def prepare_data(data_set_name = '537_houses'):
    """
    Source and prepare test and training data ready for modelling.
    :param data_set_name. The name of the data set to be downloaded and prepared.
    :return X_train, X_test, y_train, y_test. training and test test numpy arrays
    """
    # Source data
    print('Fetching the', data_set_name, 'benchmark data...')
    df = fetch_data(data_set_name)

    # Create a label
    df['big_small_yn'] = (df['total_rooms'] > 2124).astype(int)
    target_name = 'big_small_yn'
    df.drop('total_rooms', axis=1, inplace=True)
    y = df[target_name]
    df_X = df.drop(target_name, axis=1)

    # Create test and train splits
    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Calculate class imbalance
    classes = df[target_name].value_counts()
    class_0 = min(classes.index.values)
    class_1 = max(classes.index.values)
    balance_ratio = round(np.sum(df[target_name] == class_0) / float(np.sum(df[target_name] == class_1)), 2)
    return X_train, X_test, y_train, y_test, balance_ratio


def _random_search(clf, X, y, search_space, n_iter):
    """
    Random Search.
    """

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    randomsearch = RandomizedSearchCV(clf,
                                      search_space,
                                      scoring='roc_auc',
                                      cv=cv,
                                      verbose=True,
                                      n_iter=n_iter,
                                      n_jobs=-1,
                                      return_train_score=True
                                      )
    randomsearch.fit(X, y)
    return randomsearch.best_estimator_, randomsearch.best_score_, randomsearch.best_params_, randomsearch.cv_results_


def _grid_search(clf, X, y, search_space):
    """
    Grid Search.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    gridsearch = GridSearchCV(clf,
                              search_space,
                              scoring='r2',
                              cv=cv,
                              verbose=False,
                              n_jobs=-1
                              )
    gridsearch.fit(X, y)
    return gridsearch.best_estimator_, gridsearch.best_score_, gridsearch.best_params_, gridsearch.cv_results_


def tune(X_train, y_train, search_space, strategy='random', n_iter=20):
    """
    Tune a ml model using a hyper parameter optimization strategy.
    :param X_train. training features
    :param y_train. training target variable
    :param search_spaceparams.
    :param strategy. tuning strategy
    """
    # Initialise classifier with default objective function
    clf = xgb.XGBClassifier(random_state=RANDOM_SEED)
    if search_strategy == 'random':
        best_estimator, best_score, best_params, cv_results = _random_search(clf, X_train, y_train, search_space, n_iter)
    elif search_strategy == 'grid':
        best_estimator, best_score, best_params, cv_results = _grid_search(clf, X_train, y_train, search_space)
    else:
        print('Other search strategies are in development.')
    return best_estimator, best_score, best_params, cv_results


def evaulate(X_test, y_test, model):
    """
    Evaulte the tuned model on the test data set
    :param X_test. test features
    :param y_test. test target variable
    :param model. a trained sklearn model
    :return _auc, _f1, _recall. classification performance measures.
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate test set performance
    _auc = roc_auc_score(y_test, y_pred)
    _f1 = f1_score(y_test, y_pred, pos_label=1)
    _recall = recall_score(y_test, y_pred, pos_label=1)

    return _auc, _f1, _recall


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, balance_ratio = prepare_data(data_set_name)

    # Define search strategy and space
    if search_strategy == 'random':
        print ('\nInitiating a random search')
        search_space = {"learning_rate": sp_randuni(0.03, 0.25),
                    "n_estimators": sp_randint(1, X_train.shape[1]),
                    "max_depth": sp_randint(3, 10),
                    "reg_lambda": sp_randuni(0.05, 0.4),
                    "gamma": sp_randuni(0.05, 0.2),
                    "subsample": [0.5, 0.9],
                    "max_delta_step": [1.0, 1.5],
                    "scale_pos_weight": [balance_ratio]
                    }
    elif search_strategy == 'grid':
        print ('\nInitiating a grid search')
        grid =[ {"learning_rate": [0.03, 0.05]
                , "n_estimators": [1, X_train.shape[1]]
                , "subsample": [0.5, 0.9]
                , "max_delta_step": [1.0, 1.5]
                , "scale_pos_weight": [balance_ratio]
                }]
        search_space = grid
    
    # tune model
    model, best_score, best_params, cv_results = tune(X_train,
                                                      y_train,
                                                      search_space,
                                                      n_iter=10,
                                                      strategy=search_strategy)
    
    print('\nCross validation model performance (auc)')
    print(best_score)
    print('\nBest hyper parameters')
    print(best_params)
    _auc, _f1, _recall = evaulate(X_test, y_test, model)
    print('\nHold out set model performance (auc, f1, recall):')
    print(_auc, _f1, _recall)
