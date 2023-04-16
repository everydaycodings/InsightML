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


class Utils:

    def __init__(self) -> None:
        pass

    def read_file(self, raw_data):
        
        return pd.read_csv(raw_data)

    def drop_columns(self, data, selected_columns):

        return data.drop(selected_columns, axis=1)

    def target_column(self, data, selected_column):

        y = data[selected_column]
        x = data.drop(selected_column, axis = 1)

        return x, y
    

    def train_test_split(self, X, y, test_size):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test

    
    def empty_liner_dict(self):

        metrics_dict = {
            "MAE": "Not Selected",
            "MSE": "Not Selected",
            "R2": "Not Selected",
            "Adjusted R2": "Not Selected",
            "RMSE": "Not Selected",
            "Cross Validated R2": "Not Selected"
        }

        return metrics_dict



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
