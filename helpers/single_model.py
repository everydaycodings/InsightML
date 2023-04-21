import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from sklearn.ensemble  import RandomForestRegressor, RandomForestClassifier



class ApplyLinearRegression:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def Reg_Models_Evaluation_Metrics (self, model,X_train,y_train,X_test,y_test,y_pred, metrics_selector):

        cv_score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
        
        if "MAE" in metrics_selector:
            mae = round(metrics.mean_absolute_error(y_test, y_pred), 2)
        else:
            mae = None
        
        if "MSE" in metrics_selector:
            mse = round(metrics.mean_squared_error(y_test, y_pred), 2)
        else:
            mse = None
        
        if "R2" in metrics_selector:
            # Calculating Adjusted R-squared
            r2 = round(model.score(X_test, y_test), 2)
        else:
            r2 = None
        
        if "Adjusted R2" in metrics_selector:
            r2_ = model.score(X_test, y_test)
            n = X_test.shape[0]
            p = X_test.shape[1]
            adjusted_r2 = round(1-(1-r2_)*(n-1)/(n-p-1), 2)
        else:
            adjusted_r2 = None
        
        if "RMSE" in metrics_selector:
            RMSE = round(np.sqrt(mse), 2)
        else:
            RMSE = None
        
        if "Cross Validated R2" in metrics_selector:
            CV_R2 = round(cv_score.mean(), 2)
        else:
            CV_R2 = None
        

        metrics_dict = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "Adjusted R2": adjusted_r2,
            "RMSE": RMSE,
            "Cross Validated R2": CV_R2
        }

        return metrics_dict


    def apply_model(self, metrics_selector):

        regression=LinearRegression()
        regression.fit(self.X_train,self.y_train)
        y_pred = regression.predict(self.X_test)

        metrics = self.Reg_Models_Evaluation_Metrics(regression,self.X_train,self.y_train,self.X_test,self.y_test,y_pred, metrics_selector)
    
        return metrics

    
    def apply_liner_model(self, metrics_selector, linier_model, alpha, l1_ratio=None):

        if linier_model == "Lasso Regression":
            regression=Lasso(alpha=alpha)
        elif linier_model == "Ridge Regression":
            regression=Ridge(alpha=alpha)
        elif linier_model == "ElasticNet Regression":
            regression = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        
        regression.fit(self.X_train,self.y_train)
        y_pred = regression.predict(self.X_test)

        metrics = self.Reg_Models_Evaluation_Metrics(regression,self.X_train,self.y_train,self.X_test,self.y_test,y_pred, metrics_selector)

        return metrics


class ApplyLogisticRegression:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    

    def plot_metrics(self, y_pred, clf, metrics_selector):
        
        if "Confusion Metrics" in metrics_selector:
            st.subheader("Confusion Metrics")
            cm = confusion_matrix(self.y_test, y_pred, labels=clf.classes_)
            st.text(cm)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True)
            plt.xlabel("Predicted")
            plt.ylabel("Truth")

            with tempfile.TemporaryDirectory() as path:
                img_path = '{}/my_plot.png'.format(path)
                plt.savefig(img_path)
                st.image(img_path)
        

        if "ROC Curve" in metrics_selector:
            st.subheader("ROC Curve")
            roc_auc = roc_auc_score(self.y_test, y_pred)
            roc_display = RocCurveDisplay.from_predictions(self.y_test, y_pred)
            roc_display.ax_.set_title('ROC Curve (AUC = {:0.2f})'.format(roc_auc))
            with tempfile.TemporaryDirectory() as path:
                img_path = '{}/my_plot.png'.format(path)
                plt.savefig(img_path)
                st.image(img_path)
        
        if "Precision Recall Curve" in metrics_selector:
            st.subheader("Precision Recall Curve")
            plt.figure(figsize=(10, 7))
            precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred)
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            with tempfile.TemporaryDirectory() as path:
                img_path = '{}/my_plot.png'.format(path)
                plt.savefig(img_path)
                st.image(img_path)




    def apply_model(self, penalty):

        regression = LogisticRegression(penalty=penalty)
        regression.fit(self.X_train,self.y_train)
        y_pred = regression.predict(self.X_test)
        accuracy = round(accuracy_score(self.y_test, y_pred), 3)
        recall = round(recall_score(self.y_test, y_pred), 3)
        precision = round(precision_score(self.y_test, y_pred), 3)
        f1 = round(f1_score(self.y_test, y_pred), 3)
        auc = round(roc_auc_score(self.y_test, y_pred), 3)

        return accuracy, recall, precision, f1, auc, y_pred, regression



class ApplyDecisionTreeRegressor:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def Reg_Models_Evaluation_Metrics (self, model,y_pred, metrics_selector):

        cv_score = cross_val_score(estimator = model, X = self.X_train, y = self.y_train, cv = 10)
        
        if "MAE" in metrics_selector:
            mae = round(metrics.mean_absolute_error(self.y_test, y_pred), 2)
        else:
            mae = None
        
        if "MSE" in metrics_selector:
            mse = round(metrics.mean_squared_error(self.y_test, y_pred), 2)
        else:
            mse = None
        
        if "R2" in metrics_selector:
            # Calculating Adjusted R-squared
            r2 = round(model.score(self.X_test, self.y_test), 2)
        else:
            r2 = None
        
        if "Adjusted R2" in metrics_selector:
            r2_ = model.score(self.X_test, self.y_test)
            n = self.X_test.shape[0]
            p = self.X_test.shape[1]
            adjusted_r2 = round(1-(1-r2_)*(n-1)/(n-p-1), 2)
        else:
            adjusted_r2 = None
        
        if "RMSE" in metrics_selector:
            RMSE = round(np.sqrt(mse), 2)
        else:
            RMSE = None
        
        if "Cross Validated R2" in metrics_selector:
            CV_R2 = round(cv_score.mean(), 2)
        else:
            CV_R2 = None
        

        metrics_dict = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "Adjusted R2": adjusted_r2,
            "RMSE": RMSE,
            "Cross Validated R2": CV_R2
        }

        return metrics_dict
    
    def apply_model(self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, metrics_selector):
        
        reg = DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
        reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)

        metrics = self.Reg_Models_Evaluation_Metrics(reg, y_pred, metrics_selector)
        
        return metrics


    def apply_decision_tree_models(self, model_names, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, metrics_selector, bootstrap):

        if model_names == "Random Forest":
            reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap)

        reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)

        metrics = self.Reg_Models_Evaluation_Metrics(reg, y_pred, metrics_selector)
        
        return metrics

class ApplyDecisionTreeClassifier:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    

    def evaluation_metrics(self, y_pred):

        accuracy = round(accuracy_score(self.y_test, y_pred), 3)
        recall = round(recall_score(self.y_test, y_pred), 3)
        precision = round(precision_score(self.y_test, y_pred), 3)
        f1 = round(f1_score(self.y_test, y_pred), 3)
        auc = round(roc_auc_score(self.y_test, y_pred), 3)

        metrics_dict = {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1 Score": f1,
            "AUC Score": auc
        }

        return metrics_dict
    

    def apply_model(self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease):

        clf = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        metrics_dict = self.evaluation_metrics(y_pred)

        return metrics_dict
    
    def apply_decision_tree_models(self, model_names, n_estimators, criterion,  max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease, bootstrap):

        if "Random Forest" in model_names:
            clf = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap)
        
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        metrics_dict = self.evaluation_metrics(y_pred)

        return metrics_dict