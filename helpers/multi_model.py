import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression, SGDClassifier, Perceptron
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

round_off_limit = 4

class RegressionHandler:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
     
    def Evaluation_Metrics(self, model, y_pred):

        cv_score = cross_val_score(estimator = model, X = self.X_train, y = self.y_train, cv = 10)
        
        mae = round(metrics.mean_absolute_error(self.y_test, y_pred), round_off_limit)
        mse = round(metrics.mean_squared_error(self.y_test, y_pred), round_off_limit)
        r2 = round(model.score(self.X_test, self.y_test), round_off_limit)
        
        r2_ = model.score(self.X_test, self.y_test)
        n = self.X_test.shape[0]
        p = self.X_test.shape[1]
        adjusted_r2 = round(1-(1-r2_)*(n-1)/(n-p-1), round_off_limit)
        RMSE = round(np.sqrt(mse), round_off_limit)
        CV_R2 = round(cv_score.mean(), round_off_limit)
        metrics_dict = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "Adjusted R2": adjusted_r2,
            "RMSE": RMSE,
            "Cross Validated R2": CV_R2
        }

        return metrics_dict

    def apply_linear_regression(self, model_name, alpha=None, l1_ratio=None):
        
        if model_name == "Linear Regression":
            regression=LinearRegression()
        elif model_name == "Lasso Regression":
            regression=Lasso(alpha=alpha)
        elif model_name == "Ridge Regression":
            regression=Ridge(alpha=alpha)
        elif model_name == "ElasticNet Regression":
            regression = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        regression.fit(self.X_train,self.y_train)
        y_pred = regression.predict(self.X_test)

        metrics = self.Evaluation_Metrics(regression, y_pred)
    
        return metrics

    def apply_decision_tree(self, model_name, criterion=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, max_leaf_nodes=None, min_impurity_decrease=None, n_estimators=None, bootstrap=None, splitter=None, eta=None):
        
        if model_name == "Decision Tree":
            reg = DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
        elif model_name == "Random Forest":
            reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap)
        elif model_name == "XGBoost":
            reg = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, eta=eta)

        reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)

        metrics = self.Evaluation_Metrics(reg, y_pred)
        
        return metrics

    
    def apply_svr(self, kernal, degree, gamma):

        reg = SVR(kernel=kernal, gamma=gamma, degree=degree)
        reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)

        metrics = self.Evaluation_Metrics(reg, y_pred)
        
        return metrics



    def apply_model(self, metrics_dict):

        df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        return df





class ClassifierHandler:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    
    def evaluation_metrics(self, y_pred, average, multi_class):
        
        try:
            accuracy = round(accuracy_score(self.y_test, y_pred), round_off_limit)
        except:
            accuracy = "Some Error Occured"
        try:
            recall = round(recall_score(self.y_test, y_pred, average=average), round_off_limit)
        except:
            recall = "Some Error Occured"
        try:
            precision = round(precision_score(self.y_test, y_pred, average=average), round_off_limit)
        except:
            precision = "Some Error Occured"
        try:
            f1 = round(f1_score(self.y_test, y_pred, average=average), round_off_limit)
        except:
            f1 = "Some Error Occured"
        try:
            auc = round(roc_auc_score(self.y_test, y_pred, multi_class=multi_class), round_off_limit)
        except:
            auc = "Some Error Occured"

        metrics_dict = {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1 Score": f1,
            "AUC Score": auc
        }

        return metrics_dict
    
    
    def apply_linear(self, model_name, penalty, average, multi_class, loss=None, alpha=None):

        if model_name == "Logistic Regression":
            regression = LogisticRegression(penalty=penalty)
        if model_name == "SGDClassifier":
            regression = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha)

        regression.fit(self.X_train,self.y_train)
        y_pred = regression.predict(self.X_test)

        metrics = self.evaluation_metrics(y_pred, average, multi_class)

        return metrics

    def apply_decision_tree(self, model_name, criterion=None,  max_depth=None, min_samples_split=None, min_samples_leaf=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=None, splitter=None, n_estimators=None, bootstrap=None, eta=None, average=None, multi_class=None):

        if model_name == "Decision Tree":
            clf = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
        if model_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap)
        if model_name == "XGBoost":
            clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, eta=eta)
        
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        metrics = self.evaluation_metrics(y_pred, average, multi_class)

        return metrics
    
    def apply_decision_tree_v2(self, model_name, n_estimators, learning_rate, algorithm, random_state, average, multi_class):

        if model_name == "AdaBoost":
            clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)

        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        metrics = self.evaluation_metrics(y_pred, average, multi_class)

        return metrics


    def apply_svc(self, kernal, degree, gamma, average, multi_class):

        clf = SVC(kernel=kernal, gamma=gamma, degree=degree)
        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)

        metrics = self.evaluation_metrics(y_pred, average, multi_class)
        
        return metrics

    def apply_naive_bayse(self, model_name, average, multi_class):

        if model_name == "Naive Bayes":
            clf = GaussianNB()

        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        metrics = self.evaluation_metrics(y_pred, average, multi_class)
        
        return metrics

    def apply_perceptron(self, model_name, average, multi_class):

        if model_name == "Multi-Layer Perceptron":
            clf = MLPClassifier()
        if model_name == "Perceptron":
            clf = Perceptron()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        metrics = self.evaluation_metrics(y_pred, average, multi_class)
        
        return metrics

    def appply_neighbors(self, model_name, average, multi_class):

        if model_name == "K-Neighbors":
            clf = KNeighborsClassifier()
            
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        metrics = self.evaluation_metrics(y_pred, average, multi_class)
        
        return metrics

    def apply_model(self, metrics_dict):

        df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        return df