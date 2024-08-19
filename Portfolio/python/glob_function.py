# glob_function.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR


class MLUtils:
    def __init__(self):
        pass

    @staticmethod
    def load_csv(url):
        return pd.read_csv(url)

    @staticmethod
    def eda(self):
        methods = ['describe', 'head']
        attributes = ['columns', 'dtypes', 'shape']

        print(f"{'-' * 10} INFO {'-' * 50}")
        self.info()
        print('\n')

        for method in methods:
            print(f"{'-' * 10} {method.upper()} {'-' * 50}")
            print(getattr(self, method)())
            print('\n')

        for attr in attributes:
            print(f"{'-' * 10} {attr.upper()} {'-' * 50}")
            print(getattr(self, attr))
            print('\n')

        print(f"{'-' * 10} ISNULL {'-' * 50}")
        print(self.isnull().sum())
        print('\n')

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def scale_data(X_train, X_test, scaler_type='standard'):
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type should be either 'standard' or 'minmax'")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    @staticmethod
    def encode_labels(y):
        le = LabelEncoder()
        return le.fit_transform(y)

    @staticmethod
    def train_model(X_train, y_train, model_type='random_forest', **kwargs):
        if model_type == 'random_forest':
            model = RandomForestClassifier(**kwargs)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**kwargs)
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(**kwargs)
        elif model_type == 'svm':
            model = SVC(**kwargs)
        else:
            raise ValueError("model_type should be 'random_forest', 'logistic_regression', 'decision_tree', or 'svm'")

        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(y_true, y_pred, average='macro'):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        return accuracy, precision, recall, f1, cm, report

    @staticmethod
    def tune_model(X_train, y_train, model, param_grid, cv=5):
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_